"""Cosmology helpers matching the WebSky simulation (flat LCDM, Om=0.31, h=0.68)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from astropy.cosmology import FlatLambdaCDM

WEBSKY_H = 0.68
WEBSKY_OMEGA_M = 0.049 + 0.261
RHO_CRIT0_H2 = 2.775e11  # critical density today in h^2 Msun / Mpc^3

#: Cosmology used to generate the WebSky halo catalogue and maps.
WEBSKY_COSMO = FlatLambdaCDM(H0=100 * WEBSKY_H, Om0=WEBSKY_OMEGA_M, Tcmb0=0.0)


def _chi_grid(cosmo: FlatLambdaCDM) -> tuple[np.ndarray, np.ndarray]:
    """
    Tabulate comoving distance on a redshift grid for fast inversion.

    Parameters
    ----------
    cosmo : FlatLambdaCDM
        Cosmology to tabulate (grid spans 0 < z < 6).

    Returns
    -------
    z : np.ndarray
        Redshift grid.
    chi : np.ndarray
        Comoving distance in Mpc at each grid point.
    """
    return _chi_grid_cached(cosmo.H0.value, cosmo.Om0)


@lru_cache(maxsize=4)
def _chi_grid_cached(h0: float, om0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Cached comoving-distance table keyed on hashable cosmological parameters.

    Parameters
    ----------
    h0 : float
        Hubble constant in km/s/Mpc.
    om0 : float
        Matter density parameter.

    Returns
    -------
    z : np.ndarray
        Redshift grid.
    chi : np.ndarray
        Comoving distance in Mpc.
    """
    z = np.linspace(0.0, 6.0, 20000)
    return z, FlatLambdaCDM(H0=h0, Om0=om0, Tcmb0=0.0).comoving_distance(z).value


def z_of_chi(chi: np.ndarray, cosmo: FlatLambdaCDM = WEBSKY_COSMO) -> np.ndarray:
    """
    Convert comoving distance to redshift by interpolating a tabulated grid.

    Parameters
    ----------
    chi : np.ndarray
        Comoving distance in Mpc.
    cosmo : FlatLambdaCDM, default=WEBSKY_COSMO
        Cosmology.

    Returns
    -------
    z : np.ndarray
        Redshift.
    """
    zg, chig = _chi_grid(cosmo)
    return np.interp(chi, chig, zg)


def rho_crit(z: np.ndarray, cosmo: FlatLambdaCDM = WEBSKY_COSMO) -> np.ndarray:
    """
    Physical critical density.

    Parameters
    ----------
    z : np.ndarray
        Redshift.
    cosmo : FlatLambdaCDM, default=WEBSKY_COSMO
        Cosmology.

    Returns
    -------
    rho : np.ndarray
        Critical density in Msun / Mpc^3 (physical).
    """
    h = cosmo.H0.value / 100
    return RHO_CRIT0_H2 * h**2 * cosmo.efunc(z) ** 2


def rho_mean(z: np.ndarray, cosmo: FlatLambdaCDM = WEBSKY_COSMO) -> np.ndarray:
    """
    Physical mean matter density.

    Parameters
    ----------
    z : np.ndarray
        Redshift.
    cosmo : FlatLambdaCDM, default=WEBSKY_COSMO
        Cosmology.

    Returns
    -------
    rho : np.ndarray
        Mean matter density in Msun / Mpc^3 (physical).
    """
    h = cosmo.H0.value / 100
    return RHO_CRIT0_H2 * h**2 * cosmo.Om0 * (1 + np.asarray(z)) ** 3


def angular_diameter_distance(z: np.ndarray, cosmo: FlatLambdaCDM = WEBSKY_COSMO) -> np.ndarray:
    """
    Angular diameter distance using the tabulated comoving distance.

    Parameters
    ----------
    z : np.ndarray
        Redshift.
    cosmo : FlatLambdaCDM, default=WEBSKY_COSMO
        Cosmology (must be flat).

    Returns
    -------
    d_a : np.ndarray
        Angular diameter distance in Mpc.
    """
    zg, chig = _chi_grid(cosmo)
    z = np.asarray(z)
    return np.interp(z, zg, chig) / (1 + z)
