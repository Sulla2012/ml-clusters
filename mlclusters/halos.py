"""Halo mass-definition conversions and angular sizes used to scale cluster masks."""

from __future__ import annotations

import re

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from .cosmology import WEBSKY_COSMO, angular_diameter_distance, rho_crit, rho_mean

ARCMIN_PER_RAD = 180 * 60 / np.pi


def parse_mdef(mdef: str) -> tuple[float, str]:
    """
    Parse a spherical-overdensity mass definition such as ``"500c"`` or ``"200m"``.

    Parameters
    ----------
    mdef : str
        Overdensity followed by ``c`` (critical) or ``m`` (mean matter).

    Returns
    -------
    delta : float
        Overdensity.
    ref : str
        ``"c"`` or ``"m"``.
    """
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([cm])", mdef)
    if match is None:
        raise ValueError(f"Unsupported mass definition {mdef!r}")
    return float(match.group(1)), match.group(2)


def _ref_density(z: np.ndarray, mdef: str, cosmo: FlatLambdaCDM) -> tuple[float, np.ndarray]:
    """
    Return ``(delta, rho_ref(z))`` for a mass definition.

    Parameters
    ----------
    z : np.ndarray
        Redshift.
    mdef : str
        Mass definition, e.g. ``"500c"``.
    cosmo : FlatLambdaCDM
        Cosmology.

    Returns
    -------
    delta : float
        Overdensity.
    rho : np.ndarray
        Reference density in physical Msun / Mpc^3.
    """
    delta, ref = parse_mdef(mdef)
    rho = rho_crit(z, cosmo) if ref == "c" else rho_mean(z, cosmo)
    return delta, rho


def radius_from_mass(
    m: np.ndarray, z: np.ndarray, mdef: str, cosmo: FlatLambdaCDM = WEBSKY_COSMO
) -> np.ndarray:
    """
    Physical spherical-overdensity radius for a mass.

    Parameters
    ----------
    m : np.ndarray
        Mass in Msun (no factors of h).
    z : np.ndarray
        Redshift.
    mdef : str
        Mass definition of ``m``.
    cosmo : FlatLambdaCDM, default=WEBSKY_COSMO
        Cosmology.

    Returns
    -------
    r : np.ndarray
        Radius in physical Mpc.
    """
    delta, rho = _ref_density(z, mdef, cosmo)
    return (3 * np.asarray(m) / (4 * np.pi * delta * rho)) ** (1 / 3)


def concentration_duffy08(
    m200m: np.ndarray, z: np.ndarray, h: float = WEBSKY_COSMO.h
) -> np.ndarray:
    """
    Duffy et al. (2008) full-sample NFW concentration for M200m.

    Parameters
    ----------
    m200m : np.ndarray
        M200m in Msun (no factors of h).
    z : np.ndarray
        Redshift.
    h : float
        Dimensionless Hubble parameter, used to convert the pivot mass.

    Returns
    -------
    c200m : np.ndarray
        Concentration.
    """
    m_pivot = 2e12 / h
    return 10.14 * (np.asarray(m200m) / m_pivot) ** -0.081 * (1 + np.asarray(z)) ** -1.01


def _mu(x: np.ndarray) -> np.ndarray:
    """
    Dimensionless NFW enclosed mass, ``ln(1+x) - x/(1+x)``.

    Parameters
    ----------
    x : np.ndarray
        Radius in units of the scale radius.

    Returns
    -------
    mu : np.ndarray
        Enclosed-mass function.
    """
    return np.log1p(x) - x / (1 + x)


def convert_mass(
    m: np.ndarray,
    z: np.ndarray,
    mdef_in: str = "200m",
    mdef_out: str = "500c",
    c: np.ndarray | None = None,
    cosmo: FlatLambdaCDM = WEBSKY_COSMO,
    n_iter: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert between spherical-overdensity masses assuming an NFW profile.

    The NFW scale radius is fixed by ``mdef_in`` and its concentration; the new
    radius is found by vectorised bisection on ``mu(x)/x^3``, which is monotonic.

    Parameters
    ----------
    m : np.ndarray
        Input mass in Msun.
    z : np.ndarray
        Redshift.
    mdef_in : str, default="200m"
        Mass definition of ``m``.
    mdef_out : str, default="500c"
        Desired mass definition.
    c : np.ndarray, optional
        Concentration for ``mdef_in``. Defaults to Duffy et al. (2008), which is
        only defined for ``mdef_in="200m"``.
    cosmo : FlatLambdaCDM, default=WEBSKY_COSMO
        Cosmology.
    n_iter : int, default=60
        Bisection iterations (in log x).

    Returns
    -------
    m_out : np.ndarray
        Mass in the new definition, Msun.
    r_out : np.ndarray
        Radius in the new definition, physical Mpc.
    """
    m, z = np.broadcast_arrays(np.asarray(m, float), np.asarray(z, float))
    if mdef_in == mdef_out:
        return m.copy(), radius_from_mass(m, z, mdef_in, cosmo)
    if c is None:
        if mdef_in != "200m":
            raise ValueError("Default concentration is only defined for 200m.")
        c = concentration_duffy08(m, z, cosmo.h)
    c = np.broadcast_to(np.asarray(c, float), m.shape)

    rs = radius_from_mass(m, z, mdef_in, cosmo) / c
    delta, rho = _ref_density(z, mdef_out, cosmo)
    # Solve mu(x)/x^3 = target for x = r_out / rs; mu(x)/x^3 decreases with x.
    target = 4 * np.pi / 3 * delta * rho * rs**3 * _mu(c) / m
    lo, hi = np.full(m.shape, np.log(1e-3)), np.full(m.shape, np.log(1e3))
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        x = np.exp(mid)
        too_small = _mu(x) / x**3 > target
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)
    x = np.exp(0.5 * (lo + hi))
    m_out = m * _mu(x) / _mu(c)
    return m_out, x * rs


def theta_delta(
    m: np.ndarray,
    z: np.ndarray,
    mdef_in: str = "200m",
    mdef_out: str = "500c",
    cosmo: FlatLambdaCDM = WEBSKY_COSMO,
) -> np.ndarray:
    """
    Angular radius of a halo for a given overdensity definition.

    Parameters
    ----------
    m : np.ndarray
        Mass in Msun in the ``mdef_in`` definition.
    z : np.ndarray
        Redshift.
    mdef_in : str, default="200m"
        Mass definition of ``m``.
    mdef_out : str, default="500c"
        Overdensity used for the radius.
    cosmo : FlatLambdaCDM, default=WEBSKY_COSMO
        Cosmology.

    Returns
    -------
    theta : np.ndarray
        Angular radius in arcmin.
    """
    _, r = convert_mass(m, z, mdef_in, mdef_out, cosmo=cosmo)
    return r / angular_diameter_distance(z, cosmo) * ARCMIN_PER_RAD


def mask_radius(
    theta: np.ndarray,
    scale: float = 1.0,
    r_min: float = 1.0,
    r_max: float = 6.0,
) -> np.ndarray:
    """
    Mask radius from a halo's angular size, clipped to a sensible range.

    The floor keeps every mask resolvable (a few pixels at 0.5' resolution)
    and the ceiling stops nearby, low-redshift halos from swallowing a tile.

    Parameters
    ----------
    theta : np.ndarray
        Angular radius of the halo (e.g. theta500c) in arcmin.
    scale : float, default=1.0
        Multiplicative factor applied to ``theta``.
    r_min : float, default=1.0
        Minimum radius in arcmin.
    r_max : float, default=6.0
        Maximum radius in arcmin.

    Returns
    -------
    r : np.ndarray
        Mask radius in arcmin.
    """
    return np.clip(scale * np.asarray(theta), r_min, r_max)
