"""Per-tile cluster targets with mass-scaled radii: instance masks, heatmaps and ignore regions."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .catalogs import Catalog
from .geometry import TileWCS, chord, radec_to_vec
from .halos import mask_radius


@dataclass
class MaskConfig:
    """
    How catalogue clusters are turned into training targets.

    Attributes
    ----------
    m_cut : float
        M200m (Msun) above which a halo is a positive.
    m_ignore : float
        Halos with ``m_ignore <= M200m < m_cut`` are neither positive nor
        negative: their discs are excluded from the loss.
    scale : float
        Mask radius in units of theta500c.
    r_min, r_max : float
        Clip range of the mask radius in arcmin.
    sigma_frac : float
        Heatmap Gaussian width as a fraction of the mask radius.
    """

    m_cut: float = 2e14
    m_ignore: float = 1e14
    scale: float = 1.0
    r_min: float = 1.0
    r_max: float = 6.0
    sigma_frac: float = 0.5


@dataclass
class TileTargets:
    """
    Ragged per-tile cluster lists stored as flat columns plus offsets.

    Rows ``offsets[i]:offsets[i+1]`` belong to tile ``i``.

    Attributes
    ----------
    offsets : np.ndarray
        ``(n_tiles + 1,)`` row offsets.
    x, y : np.ndarray
        0-based pixel position of each cluster in its tile (may lie slightly
        outside the tile).
    theta500c : np.ndarray
        Angular radius in arcmin.
    m200m, m500c, z : np.ndarray
        Halo properties.
    cat_idx : np.ndarray
        Row in the source catalogue.
    """

    offsets: np.ndarray
    x: np.ndarray
    y: np.ndarray
    theta500c: np.ndarray
    m200m: np.ndarray
    m500c: np.ndarray
    z: np.ndarray
    cat_idx: np.ndarray

    def tile(self, i: int) -> dict[str, np.ndarray]:
        """
        Cluster columns for a single tile.

        Parameters
        ----------
        i : int
            Tile index.

        Returns
        -------
        rows : dict[str, np.ndarray]
            Column name to values for the clusters in tile ``i``.
        """
        s = slice(self.offsets[i], self.offsets[i + 1])
        return {f.name: getattr(self, f.name)[s] for f in fields(self) if f.name != "offsets"}

    def save(self, path: str | Path) -> None:
        """
        Save to a compressed ``.npz``.

        Parameters
        ----------
        path : str | Path
            Output path.
        """
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path: str | Path) -> TileTargets:
        """
        Load from ``.npz``.

        Parameters
        ----------
        path : str | Path
            Input path.

        Returns
        -------
        targets : TileTargets
            The loaded targets.
        """
        with np.load(path) as d:
            return cls(**{f.name: d[f.name] for f in fields(cls)})


def compute_tile_targets(
    wcs: TileWCS, shape: tuple[int, int], cat: Catalog, margin_arcmin: float = 7.0
) -> TileTargets:
    """
    Find every catalogue cluster in (or within a margin of) each tile.

    Parameters
    ----------
    wcs : TileWCS
        Tile geometry.
    shape : tuple[int, int]
        Tile shape ``(ny, nx)``.
    cat : Catalog
        Cluster catalogue.
    margin_arcmin : float, default=7.0
        Clusters whose centres lie up to this far outside the tile are kept so
        their discs can be ignored at the tile edges. Should be >= ``r_max``.

    Returns
    -------
    targets : TileTargets
        Per-tile cluster lists.
    """
    ny, nx = shape
    pix_arcmin = np.abs(wcs.cdelt2) * 60
    margin = margin_arcmin / pix_arcmin
    half_diag_deg = np.hypot(nx, ny) / 2 * np.abs(wcs.cdelt2).max() + margin_arcmin / 60
    ra_c, dec_c = wcs.pix2world(np.full(len(wcs), (nx - 1) / 2), np.full(len(wcs), (ny - 1) / 2))
    tree = cKDTree(radec_to_vec(cat.ra, cat.dec))
    cands = tree.query_ball_point(radec_to_vec(ra_c, dec_c), chord(half_diag_deg))
    tile_idx = np.repeat(np.arange(len(wcs)), [len(c) for c in cands])
    cat_idx = np.concatenate([np.asarray(c, int) for c in cands] + [np.zeros(0, int)])
    x, y = wcs.world2pix(cat.ra[cat_idx], cat.dec[cat_idx], tile_idx)
    m = margin[tile_idx]
    keep = (x > -m) & (x < nx - 1 + m) & (y > -m) & (y < ny - 1 + m)
    tile_idx, cat_idx, x, y = tile_idx[keep], cat_idx[keep], x[keep], y[keep]
    order = np.lexsort((cat_idx, tile_idx))
    tile_idx, cat_idx, x, y = tile_idx[order], cat_idx[order], x[order], y[order]
    offsets = np.concatenate([[0], np.cumsum(np.bincount(tile_idx, minlength=len(wcs)))])
    return TileTargets(
        offsets=offsets,
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        theta500c=cat.theta500c[cat_idx].astype(np.float32),
        m200m=cat.m200m[cat_idx],
        m500c=cat.m500c[cat_idx],
        z=cat.z[cat_idx].astype(np.float32),
        cat_idx=cat_idx,
    )


def radius_pix(theta500c: np.ndarray, pix_arcmin: float, cfg: MaskConfig) -> np.ndarray:
    """
    Mass-scaled mask radius in pixels.

    Parameters
    ----------
    theta500c : np.ndarray
        Angular radius in arcmin.
    pix_arcmin : float
        Pixel size in arcmin.
    cfg : MaskConfig
        Radius scaling configuration.

    Returns
    -------
    r : np.ndarray
        Radius in pixels.
    """
    return mask_radius(theta500c, cfg.scale, cfg.r_min, cfg.r_max) / pix_arcmin


def _grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Pixel-centre coordinate grids.

    Parameters
    ----------
    shape : tuple[int, int]
        ``(ny, nx)``.

    Returns
    -------
    yy, xx : np.ndarray
        Broadcastable ``(ny, 1)`` and ``(1, nx)`` grids.
    """
    return np.arange(shape[0])[:, None], np.arange(shape[1])[None, :]


def make_mask(shape: tuple[int, int], x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Instance mask with one disc per cluster, each with its own radius.

    Overlapping pixels go to the cluster with the smallest normalised distance
    ``d / r``, so a small cluster next to a large one keeps its core.

    Parameters
    ----------
    shape : tuple[int, int]
        ``(ny, nx)``.
    x, y : np.ndarray
        0-based cluster centres in pixels.
    r : np.ndarray
        Radius of each disc in pixels (scalars are broadcast).

    Returns
    -------
    mask : np.ndarray
        ``int32`` image; 0 is background and ``k + 1`` marks cluster ``k``.
    """
    x, y, r = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, float)) for v in (x, y, r)))
    mask = np.zeros(shape, np.int32)
    best = np.full(shape, np.inf)
    yy, xx = _grid(shape)
    for k in range(len(x)):
        d = np.hypot(xx - x[k], yy - y[k]) / r[k]
        win = (d <= 1) & (d < best)
        mask[win] = k + 1
        best = np.where(win, d, best)
    return mask


def make_heatmap(
    shape: tuple[int, int], x: np.ndarray, y: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """
    CenterNet-style heatmap: element-wise max of per-cluster Gaussians.

    The pixel nearest each centre is set to exactly 1 so it is a positive.

    Parameters
    ----------
    shape : tuple[int, int]
        ``(ny, nx)``.
    x, y : np.ndarray
        0-based centres (must lie inside the tile).
    sigma : np.ndarray
        Gaussian widths in pixels.

    Returns
    -------
    heat : np.ndarray
        ``float32`` heatmap in [0, 1].
    """
    heat = np.zeros(shape, np.float32)
    yy, xx = _grid(shape)
    for xk, yk, sk in zip(x, y, sigma, strict=True):
        g = np.exp(-((xx - xk) ** 2 + (yy - yk) ** 2) / (2 * sk**2))
        np.maximum(heat, g, out=heat)
        heat[int(round(yk)), int(round(xk))] = 1.0
    return heat


def tile_training_maps(
    rows: dict[str, np.ndarray], shape: tuple[int, int], pix_arcmin: float, cfg: MaskConfig
) -> dict[str, np.ndarray]:
    """
    Build the dense training targets for one tile.

    Parameters
    ----------
    rows : dict[str, np.ndarray]
        Output of :meth:`TileTargets.tile`.
    shape : tuple[int, int]
        ``(ny, nx)``.
    pix_arcmin : float
        Pixel size in arcmin.
    cfg : MaskConfig
        Target configuration.

    Returns
    -------
    maps : dict[str, np.ndarray]
        ``heat`` (centre heatmap), ``radius`` (log mask radius in pixels at
        centre pixels), ``reg_mask`` (1 at centre pixels) and ``weight``
        (0 inside ignore regions, 1 elsewhere).
    """
    ny, nx = shape
    r = radius_pix(rows["theta500c"], pix_arcmin, cfg)
    x, y, m = rows["x"], rows["y"], rows["m200m"]
    inside = (x > -0.5) & (x < nx - 0.5) & (y > -0.5) & (y < ny - 0.5)
    pos = inside & (m >= cfg.m_cut)
    ignore = ~pos & (m >= cfg.m_ignore)  # sub-threshold halos and off-tile positives

    heat = make_heatmap(shape, x[pos], y[pos], np.maximum(cfg.sigma_frac * r[pos], 0.5))
    radius = np.zeros(shape, np.float32)
    reg_mask = np.zeros(shape, np.float32)
    iy, ix = np.round(y[pos]).astype(int), np.round(x[pos]).astype(int)
    radius[iy, ix] = np.log(r[pos])
    reg_mask[iy, ix] = 1.0
    weight = (make_mask(shape, x[ignore], y[ignore], r[ignore]) == 0).astype(np.float32)
    weight = np.maximum(weight, heat > 0.5)  # never ignore a positive's core
    return {"heat": heat, "radius": radius, "reg_mask": reg_mask, "weight": weight}
