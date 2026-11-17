"""Sky geometry: gnomonic tile projections, sky matching and spatial train/test splits."""

from __future__ import annotations

from dataclasses import dataclass

import healpy as hp
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

DEG = np.pi / 180


@dataclass
class TileWCS:
    """
    Vectorised gnomonic (RA---TAN/DEC--TAN, north-up) WCS for a batch of tiles.

    All arrays have one entry per tile. Pixel coordinates are 0-based with
    ``x`` along the last (RA) axis and ``y`` along the second-to-last axis.

    Attributes
    ----------
    crval1, crval2 : np.ndarray
        Tangent point RA/Dec in degrees.
    crpix1, crpix2 : np.ndarray
        1-based FITS reference pixel.
    cdelt1, cdelt2 : np.ndarray
        Pixel scale in degrees (``cdelt1`` is usually negative).
    """

    crval1: np.ndarray
    crval2: np.ndarray
    crpix1: np.ndarray
    crpix2: np.ndarray
    cdelt1: np.ndarray
    cdelt2: np.ndarray

    @classmethod
    def from_headers(cls, headers: list[fits.Header]) -> TileWCS:
        """
        Build from FITS headers of TAN tiles.

        Parameters
        ----------
        headers : list[fits.Header]
            Headers with CRVAL/CRPIX/CDELT keywords and a TAN projection.

        Returns
        -------
        wcs : TileWCS
            Batched WCS.
        """
        for h in headers:
            if not h["CTYPE1"].endswith("TAN") or h.get("PC1_2", 0) != 0:
                raise ValueError("Only un-rotated TAN tiles are supported")
        keys = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2"]
        return cls(*[np.array([h[k] for h in headers], float) for k in keys])

    def __getitem__(self, sel: np.ndarray | slice | int) -> TileWCS:
        """
        Select a subset of tiles.

        Parameters
        ----------
        sel : np.ndarray | slice | int
            Selection.

        Returns
        -------
        wcs : TileWCS
            Selected tiles (always array-valued).
        """
        return TileWCS(*[np.atleast_1d(v[sel]) for v in self.as_dict().values()])

    def __len__(self) -> int:
        """
        Number of tiles.

        Returns
        -------
        n : int
            Number of tiles.
        """
        return len(self.crval1)

    def as_dict(self) -> dict[str, np.ndarray]:
        """
        Arrays keyed by attribute name (for saving).

        Returns
        -------
        d : dict[str, np.ndarray]
            Attribute arrays.
        """
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def world2pix(
        self, ra: np.ndarray, dec: np.ndarray, idx: np.ndarray | int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project sky positions into tile pixel coordinates.

        Parameters
        ----------
        ra, dec : np.ndarray
            Coordinates in degrees.
        idx : np.ndarray | int, optional
            Tile index for each position; broadcast against ``ra``.
            Defaults to all tiles (requires matching lengths).

        Returns
        -------
        x, y : np.ndarray
            0-based pixel coordinates; NaN for points on the far hemisphere.
        """
        idx = slice(None) if idx is None else idx
        a0, d0 = self.crval1[idx] * DEG, self.crval2[idx] * DEG
        a, d = np.asarray(ra) * DEG, np.asarray(dec) * DEG
        cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = np.cos(d) * np.sin(a - a0) / cosc / DEG
            eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc / DEG
        far = cosc <= 0
        x = np.where(far, np.nan, self.crpix1[idx] - 1 + xi / self.cdelt1[idx])
        y = np.where(far, np.nan, self.crpix2[idx] - 1 + eta / self.cdelt2[idx])
        return x, y

    def pix2world(
        self, x: np.ndarray, y: np.ndarray, idx: np.ndarray | int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert tile pixel coordinates to sky positions.

        Parameters
        ----------
        x, y : np.ndarray
            0-based pixel coordinates.
        idx : np.ndarray | int, optional
            Tile index for each position. Defaults to all tiles.

        Returns
        -------
        ra, dec : np.ndarray
            Coordinates in degrees, ``ra`` in [0, 360).
        """
        idx = slice(None) if idx is None else idx
        a0, d0 = self.crval1[idx] * DEG, self.crval2[idx] * DEG
        xi = (np.asarray(x) - self.crpix1[idx] + 1) * self.cdelt1[idx] * DEG
        eta = (np.asarray(y) - self.crpix2[idx] + 1) * self.cdelt2[idx] * DEG
        rho = np.hypot(xi, eta)
        c = np.arctan(rho)
        with np.errstate(divide="ignore", invalid="ignore"):
            sin_term = np.where(rho > 0, eta * np.sin(c) * np.cos(d0) / rho, 0.0)
        dec = np.arcsin(np.cos(c) * np.sin(d0) + sin_term)
        ra = a0 + np.arctan2(
            xi * np.sin(c), rho * np.cos(d0) * np.cos(c) - eta * np.sin(d0) * np.sin(c)
        )
        return (ra / DEG) % 360, dec / DEG


def radec_to_vec(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """
    Convert RA/Dec to unit vectors.

    Parameters
    ----------
    ra, dec : np.ndarray
        Coordinates in degrees.

    Returns
    -------
    vec : np.ndarray
        Array of shape ``(n, 3)``.
    """
    return np.asarray(hp.ang2vec(np.asarray(ra), np.asarray(dec), lonlat=True)).reshape(-1, 3)


def chord(theta_deg: float) -> float:
    """
    Chord length on the unit sphere for an angular separation.

    Parameters
    ----------
    theta_deg : float
        Separation in degrees.

    Returns
    -------
    d : float
        Chord length.
    """
    return 2 * np.sin(theta_deg * DEG / 2)


def match_catalogs(
    ra1: np.ndarray,
    dec1: np.ndarray,
    ra2: np.ndarray,
    dec2: np.ndarray,
    tol_arcmin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Nearest-neighbour match of catalogue 1 onto catalogue 2 within a tolerance.

    Parameters
    ----------
    ra1, dec1 : np.ndarray
        Coordinates (deg) of the catalogue to match from.
    ra2, dec2 : np.ndarray
        Coordinates (deg) of the reference catalogue.
    tol_arcmin : float
        Maximum separation in arcmin.

    Returns
    -------
    idx : np.ndarray
        Index into catalogue 2 of the nearest neighbour, or -1 if none in range.
    sep : np.ndarray
        Separation in arcmin (inf if unmatched).
    """
    if len(ra1) == 0 or len(ra2) == 0:
        return np.full(len(ra1), -1), np.full(len(ra1), np.inf)
    tree = cKDTree(radec_to_vec(ra2, dec2))
    dist, idx = tree.query(radec_to_vec(ra1, dec1), distance_upper_bound=chord(tol_arcmin / 60))
    found = np.isfinite(dist)
    sep = np.full(len(ra1), np.inf)
    sep[found] = 2 * np.arcsin(dist[found] / 2) / DEG * 60
    return np.where(found, idx, -1), sep


def one_to_one_match(
    ra_pred: np.ndarray,
    dec_pred: np.ndarray,
    score: np.ndarray,
    ra_true: np.ndarray,
    dec_true: np.ndarray,
    tol_arcmin: float,
) -> np.ndarray:
    """
    Greedy one-to-one matching of predictions to truths in decreasing score order.

    Each truth can be claimed by at most one prediction, so duplicate detections
    of the same cluster count as false positives rather than inflating
    completeness.

    Parameters
    ----------
    ra_pred, dec_pred : np.ndarray
        Predicted positions in degrees.
    score : np.ndarray
        Detection scores (higher claims first).
    ra_true, dec_true : np.ndarray
        True positions in degrees.
    tol_arcmin : float
        Matching radius in arcmin.

    Returns
    -------
    match : np.ndarray
        For each prediction, the index of the matched truth or -1.
    """
    match = np.full(len(ra_pred), -1)
    if len(ra_pred) == 0 or len(ra_true) == 0:
        return match
    tree = cKDTree(radec_to_vec(ra_true, dec_true))
    cands = tree.query_ball_point(radec_to_vec(ra_pred, dec_pred), chord(tol_arcmin / 60))
    vt = radec_to_vec(ra_true, dec_true)
    vp = radec_to_vec(ra_pred, dec_pred)
    taken = np.zeros(len(ra_true), bool)
    for i in np.argsort(-score, kind="stable"):
        c = np.array(cands[i], int)
        c = c[~taken[c]]
        if len(c):
            j = c[np.argmax(vt[c] @ vp[i])]
            match[i] = j
            taken[j] = True
    return match


def dedupe(ra: np.ndarray, dec: np.ndarray, score: np.ndarray, tol_arcmin: float) -> np.ndarray:
    """
    Non-maximum suppression on the sky: keep the highest-scoring detection per group.

    Parameters
    ----------
    ra, dec : np.ndarray
        Detection positions in degrees.
    score : np.ndarray
        Detection scores.
    tol_arcmin : float
        Detections closer than this to a higher-scoring kept detection are dropped.

    Returns
    -------
    keep : np.ndarray
        Indices of retained detections, in decreasing score order.
    """
    if len(ra) == 0:
        return np.zeros(0, int)
    vec = radec_to_vec(ra, dec)
    neigh = cKDTree(vec).query_ball_point(vec, chord(tol_arcmin / 60))
    suppressed = np.zeros(len(ra), bool)
    keep = []
    for i in np.argsort(-score, kind="stable"):
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed[neigh[i]] = True
    return np.array(keep, int)


def healpix_split(
    ra: np.ndarray,
    dec: np.ndarray,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    nside: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign positions to train/val/test by HEALPix region to avoid sky overlap leaks.

    Parameters
    ----------
    ra, dec : np.ndarray
        Positions in degrees (e.g. tile centres).
    fractions : tuple[float, float, float], default=(0.8, 0.1, 0.1)
        Approximate fractions of HEALPix pixels for train/val/test.
    nside : int, default=4
        HEALPix resolution of the regions (nside=4 gives ~215 deg^2 regions).
    seed : int, default=0
        Random seed for assigning regions.

    Returns
    -------
    split : np.ndarray
        0 (train), 1 (val) or 2 (test) for each position.
    region_split : np.ndarray
        Split assigned to each HEALPix region (length ``12 * nside**2``).
    """
    npix = hp.nside2npix(nside)
    perm = np.random.default_rng(seed).permutation(npix)
    edges = np.round(np.cumsum(fractions) / np.sum(fractions) * npix).astype(int)
    region_split = np.zeros(npix, int)
    region_split[perm[edges[0] : edges[1]]] = 1
    region_split[perm[edges[1] :]] = 2
    return region_split[hp.ang2pix(nside, ra, dec, lonlat=True)], region_split


def tile_split(
    wcs: TileWCS,
    shape: tuple[int, int],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    nside: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """
    Leak-free spatial split of tiles.

    A tile is assigned to a split only if its centre and four corners all fall in
    regions of that split; tiles straddling a region boundary get ``-1`` and are
    excluded, so no sky pixel appears in two splits.

    Parameters
    ----------
    wcs : TileWCS
        Tile geometry.
    shape : tuple[int, int]
        Tile shape ``(ny, nx)``.
    fractions : tuple[float, float, float], default=(0.8, 0.1, 0.1)
        Approximate train/val/test sky fractions.
    nside : int, default=4
        HEALPix resolution of the regions.
    seed : int, default=0
        Random seed.

    Returns
    -------
    split : np.ndarray
        0/1/2 for train/val/test, -1 for boundary tiles.
    """
    ny, nx = shape
    pts = [((nx - 1) / 2, (ny - 1) / 2), (0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)]
    _, region_split = healpix_split(np.zeros(1), np.zeros(1), fractions, nside, seed)
    labels = []
    for px, py in pts:
        ra, dec = wcs.pix2world(np.full(len(wcs), px), np.full(len(wcs), py))
        labels.append(region_split[hp.ang2pix(nside, ra, dec, lonlat=True)])
    labels = np.stack(labels)
    return np.where(np.all(labels == labels[0], axis=0), labels[0], -1)
