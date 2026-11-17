"""Build WebSky sky-model maps (CMB + kSZ + tSZ + CIB + radio [+ white noise]) in uK_CMB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import numpy as np
from pixell import curvedsky, enmap, reproject, utils, wcsutils

T_CMB = 2.726  # K
H_PLANCK, K_BOLTZ, C_LIGHT = 6.626068e-34, 1.3806503e-23, 299792458.0


@dataclass(frozen=True)
class Band:
    """
    Component files and conversion factors for one frequency band.

    Attributes
    ----------
    cib : str
        CIB intensity map (MJy/sr) closest to the band.
    radio : str
        Radio source map (Jy/sr).
    tsz_factor : float
        uK_CMB per unit Compton-y at the band's effective tSZ frequency.
    cib_factor : float
        uK_CMB per MJy/sr at the band's effective CIB frequency.
    radio_ghz : float
        Frequency used to convert the radio map to temperature.
    """

    cib: str
    radio: str
    tsz_factor: float
    cib_factor: float
    radio_ghz: float


#: ACT-like bands. NOTE: the previous code had the 90/150 CIB factors swapped, and the
#: f150 map on disk (2024) was built with the 90 GHz CIB map; both are fixed here.
BANDS = {
    "090": Band("cib_90.2_car.fits", "map_radio_0.5arcmin_f90.2.fits", -4.2840e6, 4.6831e3, 90.2),
    "150": Band("cib_153_car.fits", "map_radio_0.5arcmin_f143.0.fits", -2.7685e6, 2.5947e3, 143.0),
    "220": Band("cib_217_car.fits", "map_radio_0.5arcmin_f217.0.fits", 3.1517e5, 2.0676e3, 217.0),
}


def dbnu_dt(nu_ghz: float) -> float:
    """
    Derivative of the blackbody intensity with temperature at T_CMB.

    Parameters
    ----------
    nu_ghz : float
        Frequency in GHz.

    Returns
    -------
    d : float
        dB/dT in Jy/sr per uK.
    """
    nu = 1e9 * nu_ghz
    x = H_PLANCK * nu / (K_BOLTZ * T_CMB)
    dbdt_si = 2 * H_PLANCK * nu**3 / C_LIGHT**2 * np.exp(x) / np.expm1(x) ** 2 * x / T_CMB
    return dbdt_si * 1e26 / 1e6  # W/m^2/Hz/sr/K -> Jy/sr/uK


def _healpix_to_car(path: Path, shape: tuple[int, ...], wcs: object) -> enmap.ndmap:
    """
    Project a HEALPix map onto a CAR geometry harmonically.

    Parameters
    ----------
    path : Path
        HEALPix FITS map.
    shape : tuple[int, ...]
        Output shape.
    wcs : object
        Output WCS.

    Returns
    -------
    omap : enmap.ndmap
        Projected map.
    """
    hmap = hp.read_map(path)
    return reproject.healpix2map(hmap, shape=shape, wcs=wcs, lmax=3 * hp.npix2nside(hmap.size))


def _car_on_geometry(path: Path, shape: tuple[int, ...], wcs: object) -> enmap.ndmap:
    """
    Read a CAR map and put it on the output geometry.

    Maps already on the output pixelisation are used as is; others are
    interpolated by WCS (``enmap.project``).

    Parameters
    ----------
    path : Path
        CAR FITS map.
    shape : tuple[int, ...]
        Output shape.
    wcs : object
        Output WCS.

    Returns
    -------
    omap : enmap.ndmap
        Map on ``(shape, wcs)``.
    """
    imap = enmap.read_map(str(path))
    if imap.shape[-2:] == tuple(shape[-2:]) and wcsutils.equal(imap.wcs, wcs):
        return imap
    return enmap.project(imap, shape, wcs)


def make_websky_map(
    band: str,
    root: Path,
    res_arcmin: float = 0.5,
    noise_uk_arcmin: float | None = None,
    seed: int | None = None,
) -> enmap.ndmap:
    """
    Sum the WebSky components into a full-sky temperature map for one band.

    Parameters
    ----------
    band : str
        One of :data:`BANDS` (``"090"``, ``"150"``, ``"220"``).
    root : Path
        Directory with the WebSky component files.
    res_arcmin : float, default=0.5
        Output CAR resolution.
    noise_uk_arcmin : float, optional
        White-noise level; no noise if None.
    seed : int, optional
        Noise seed.

    Returns
    -------
    omap : enmap.ndmap
        Map in uK_CMB.
    """
    b = BANDS[band]
    shape, wcs = enmap.fullsky_geometry(res=res_arcmin * utils.arcmin)
    alm = hp.read_alm(root / "lensed_alm.fits", hdu=1).astype(np.complex128)
    omap = curvedsky.alm2map(alm, enmap.zeros(shape, wcs, np.float64))
    print("added CMB", flush=True)
    omap += _healpix_to_car(root / "ksz.fits", shape, wcs)
    print("added kSZ", flush=True)
    omap += b.tsz_factor * _healpix_to_car(root / "tsz_8192.fits", shape, wcs)
    print("added tSZ", flush=True)
    omap += b.cib_factor * _car_on_geometry(root / b.cib, shape, wcs)
    print(f"added CIB ({b.cib})", flush=True)
    omap += _car_on_geometry(root / b.radio, shape, wcs) / dbnu_dt(b.radio_ghz)
    print(f"added radio ({b.radio})", flush=True)
    if noise_uk_arcmin:
        rng = np.random.default_rng(seed)
        pix_arcmin = np.sqrt(enmap.pixsizemap(shape, wcs)) / utils.arcmin
        omap += rng.standard_normal(shape) * noise_uk_arcmin / pix_arcmin
    return omap
