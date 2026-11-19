"""Instrument model for simulated maps: ACT beams and ivar-based, atmosphere-coloured noise.

Noise model: ``n(x) = w(x) / sqrt(ivar(x)) + d(x) * r(x)``, where ``w`` is unit white noise,
``d = sqrt(pixel_area / ivar)`` the local depth [uK arcmin] and ``r`` a stationary Gaussian
field with power ``arcmin^2 * (S(ell) - 1)``: the atmospheric part of a 1 uK-arcmin map. ``S`` is
the measured noise power relative to white (:class:`NoiseShape`), so the total power is
``depth^2 S(ell)``, scaled to the local depth.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pixell import curvedsky, enmap, utils

ACT_DR6 = Path("/mnt/welch/ACT/20240323_simple")


@dataclass
class NoiseShape:
    """
    Noise power relative to its white level, tabulated from data.

    Attributes
    ----------
    ell : list[float]
        Multipoles of the table (increasing).
    rel : list[float]
        Power relative to white at ``ell``. Interpolated in log-log; held constant below the
        first entry and above the last.
    """

    ell: list[float]
    rel: list[float]

    def __call__(self, ell: np.ndarray) -> np.ndarray:
        """
        Total power relative to white (at least 1).

        Parameters
        ----------
        ell : np.ndarray
            Multipoles (> 0).

        Returns
        -------
        shape : np.ndarray
            Relative power.
        """
        log_rel = np.interp(np.log(np.maximum(ell, 1.0)), np.log(self.ell), np.log(self.rel))
        return np.maximum(np.exp(log_rel), 1.0)

    def save(self, path: Path, **extra: object) -> None:
        """
        Write the table (and extra metadata) to JSON.

        Parameters
        ----------
        path : Path
            Output file.
        **extra : object
            Extra JSON-serialisable entries.
        """
        Path(path).write_text(json.dumps({**asdict(self), **extra}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "NoiseShape":
        """
        Read a table written by :meth:`save`.

        Parameters
        ----------
        path : Path
            JSON file.

        Returns
        -------
        shape : NoiseShape
            Noise shape.
        """
        d = json.loads(Path(path).read_text())
        return cls(d["ell"], d["rel"])


def act_files(band: str, kind: str, coadd: str = "daynight", srcfree: bool = False) -> Path:
    """
    Path to an ACT DR6 (ACT-only, s08-s22; s17-s22 at f220) product.

    Parameters
    ----------
    band : str
        ``"090"``, ``"150"`` or ``"220"``.
    kind : str
        ``"map"``, ``"ivar"`` or ``"beam"``.
    coadd : str, default="daynight"
        ``"daynight"`` or ``"night"``.
    srcfree : bool, default=False
        Point-source-subtracted map.

    Returns
    -------
    path : Path
        File path.
    """
    if kind == "beam":
        return ACT_DR6 / f"beam_f{band}_tform.txt"
    seasons = "s17_s22" if band == "220" else "s08_s22"
    suffix = "_srcfree" if srcfree and kind == "map" else ""
    return ACT_DR6 / f"act_{seasons}_f{band}_{coadd}_{kind}{suffix}.fits"


def load_beam(path: Path, lmax: int) -> np.ndarray:
    """
    Read an ACT beam transform and normalise it to B(0) = 1.

    Parameters
    ----------
    path : Path
        ``beam_f*_tform.txt`` (columns: ell, B_ell, error modes).
    lmax : int
        Highest multipole returned.

    Returns
    -------
    bl : np.ndarray
        Beam transform for ell = 0..lmax.
    """
    return utils.regularize_beam(np.loadtxt(path).T[1], nl=lmax + 1, normalize=True)


def beam_fwhm_arcmin(bl: np.ndarray) -> float:
    """
    FWHM of the Gaussian with the same ell where B_ell = 1/2.

    Parameters
    ----------
    bl : np.ndarray
        Beam transform, B(0) = 1.

    Returns
    -------
    fwhm : float
        Equivalent Gaussian FWHM [arcmin].
    """
    if not np.any(bl < 0.5):
        return float("nan")
    ell_half = np.argmax(bl < 0.5)
    sigma = np.sqrt(2 * np.log(2)) / ell_half
    return sigma * np.sqrt(8 * np.log(2)) / utils.arcmin


def convolve_beam(imap: enmap.ndmap, bl: np.ndarray) -> enmap.ndmap:
    """
    Convolve a full-sky CAR map with a circular beam (spherical harmonics).

    Parameters
    ----------
    imap : enmap.ndmap
        Input map.
    bl : np.ndarray
        Beam transform; its length sets lmax.

    Returns
    -------
    omap : enmap.ndmap
        Beam-convolved map (float64).
    """
    ainfo = curvedsky.alm_info(len(bl) - 1)
    alm = curvedsky.map2alm(imap, ainfo=ainfo)
    ainfo.lmul(alm, bl, out=alm)
    return curvedsky.alm2map(alm, enmap.zeros(imap.shape, imap.wcs, np.float64))


def read_ivar(path: Path, shape: tuple[int, ...], wcs: object) -> enmap.ndmap:
    """
    Read the temperature ivar and put it on another CAR geometry of the same pixel size.

    Uses the nearest input pixel along each (separable) CAR axis, wrapping in RA, so the
    footprint is preserved exactly up to the half-pixel offset between pixelisations.

    Parameters
    ----------
    path : Path
        ACT ivar map (``(3, ny, nx)`` or ``(ny, nx)``) [1/uK^2 per pixel].
    shape : tuple[int, ...]
        Output shape.
    wcs : object
        Output CAR WCS (zero outside the input footprint).

    Returns
    -------
    ivar : enmap.ndmap
        Ivar on the output geometry.
    """
    ishape, _ = enmap.read_map_geometry(str(path))
    ivar = enmap.read_map(str(path), sel=np.s_[0] if len(ishape) == 3 else None)
    ny, nx = shape[-2:]
    dec = enmap.pix2sky(shape, wcs, [np.arange(ny), np.zeros(ny)])[0]
    ra = enmap.pix2sky(shape, wcs, [np.zeros(nx), np.arange(nx)])[1]
    iy = np.round(ivar.sky2pix([dec, np.zeros(ny)])[0]).astype(int)
    ix = np.round(ivar.sky2pix([np.zeros(nx), ra])[1]).astype(int)
    nwrap = int(round(2 * np.pi / np.abs(ivar.wcs.wcs.cdelt[0] * utils.degree)))
    ix %= nwrap
    out = enmap.zeros(shape[-2:], wcs, np.float32)
    oky = (iy >= 0) & (iy < ivar.shape[-2])
    okx = ix < ivar.shape[-1]
    out[np.ix_(oky, okx)] = ivar[np.ix_(iy[oky], ix[okx])]
    return out


def depth_map(ivar: enmap.ndmap) -> enmap.ndmap:
    """
    Local white-noise depth.

    Parameters
    ----------
    ivar : enmap.ndmap
        Inverse variance per pixel [1/uK^2].

    Returns
    -------
    depth : enmap.ndmap
        Depth [uK arcmin]; 0 where ivar is 0.
    """
    area = enmap.pixsizemap(ivar.shape, ivar.wcs) / utils.arcmin**2
    with np.errstate(divide="ignore"):
        return enmap.samewcs(np.where(ivar > 0, np.sqrt(area / ivar), 0.0), ivar)


def noise_map(
    ivar: enmap.ndmap, shape: NoiseShape, lmax_red: int = 8000, seed: int | None = None
) -> enmap.ndmap:
    """
    One noise realisation: inhomogeneous white noise plus depth-scaled atmospheric noise.

    Parameters
    ----------
    ivar : enmap.ndmap
        Inverse variance per pixel [1/uK^2] on the output geometry.
    shape : NoiseShape
        Relative noise spectrum.
    lmax_red : int, default=8000
        Band limit of the atmospheric part (its power is negligible above this).
    seed : int, optional
        Random seed.

    Returns
    -------
    noise : enmap.ndmap
        Noise [uK]; 0 outside the footprint.
    """
    rng = np.random.default_rng(seed)
    ps_red = utils.arcmin**2 * (shape(np.arange(lmax_red + 1)) - 1.0)
    ps_red[:2] = 0.0
    red = curvedsky.rand_map(
        ivar.shape, ivar.wcs, ps_red, lmax=lmax_red, seed=int(rng.integers(2**31))
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        white = np.where(ivar > 0, rng.standard_normal(ivar.shape) / np.sqrt(ivar), 0.0)
    return enmap.samewcs(white + depth_map(ivar) * red, ivar)


def patch_power(
    imap: enmap.ndmap, weight: enmap.ndmap, bins: np.ndarray, apod_deg: float = 1.0
) -> np.ndarray:
    """
    Binned flat-sky power spectrum of a weighted, apodised patch.

    Parameters
    ----------
    imap : enmap.ndmap
        Patch.
    weight : enmap.ndmap
        Pixel weights (0 = masked); applied before apodisation.
    bins : np.ndarray
        Multipole bin edges.
    apod_deg : float, default=1.0
        Apodisation width at the patch edges [deg].

    Returns
    -------
    ps : np.ndarray
        Mean power per bin [uK^2 sr], corrected for the mask/apodisation (w2 factor).
    """
    apod = enmap.apod(
        enmap.ones(imap.shape, imap.wcs),
        int(apod_deg * utils.degree / np.abs(imap.wcs.wcs.cdelt[1] * utils.degree)),
    )
    win = weight * apod
    fmap = enmap.fft(imap * win, normalize="phys")
    p2d = np.abs(fmap) ** 2 / np.mean(win**2)
    lmap = imap.modlmap()
    idx = np.digitize(lmap.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < len(bins) - 1)
    return np.bincount(idx[ok], p2d.ravel()[ok], len(bins) - 1) / np.maximum(
        np.bincount(idx[ok], minlength=len(bins) - 1), 1
    )
