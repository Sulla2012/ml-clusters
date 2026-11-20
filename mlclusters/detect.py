"""Run a detector over tiles, build a sky catalogue and score it against the truth."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import healpy as hp
import numpy as np
import torch

from .catalogs import Catalog
from .data import PackedTiles
from .geometry import dedupe, match_catalogs, one_to_one_match
from .masks import MaskConfig

#: A detector maps a batch of images to per-image dicts with x, y, score (and radius).
Detector = Callable[[torch.Tensor], list[dict[str, torch.Tensor]]]


@dataclass
class Detections:
    """
    Flat table of detections.

    Attributes
    ----------
    tile : np.ndarray
        Packed tile index of each detection.
    x, y : np.ndarray
        0-based pixel position in its tile.
    score : np.ndarray
        Detection score.
    radius : np.ndarray
        Predicted radius in pixels (NaN if the detector has none).
    m200m : np.ndarray
        Predicted M200m in Msun (NaN if the detector has no mass head).
    ra, dec : np.ndarray
        Sky position in degrees (filled by :func:`to_sky`).
    """

    tile: np.ndarray
    x: np.ndarray
    y: np.ndarray
    score: np.ndarray
    radius: np.ndarray
    m200m: np.ndarray
    ra: np.ndarray | None = None
    dec: np.ndarray | None = None

    def select(self, sel: np.ndarray) -> Detections:
        """
        Subset of detections.

        Parameters
        ----------
        sel : np.ndarray
            Boolean mask or indices.

        Returns
        -------
        dets : Detections
            Selected rows.
        """
        return Detections(**{k: None if v is None else v[sel] for k, v in self.__dict__.items()})


@torch.no_grad()
def predict(
    detector: Detector,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Detections:
    """
    Run a detector over every tile served by a loader.

    Parameters
    ----------
    detector : Detector
        Callable producing per-image detections for a batch of images.
    loader : torch.utils.data.DataLoader
        Yields dicts with ``image`` and ``index`` (packed tile index).
    device : torch.device
        Device for inference.

    Returns
    -------
    dets : Detections
        All detections.
    """
    keys = ("tile", "x", "y", "score", "radius", "m200m")
    cols: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for batch in loader:
        dets = detector(batch["image"].to(device, non_blocking=True))
        for t, d in zip(batch["index"].numpy(), dets, strict=True):
            n = len(d["score"])
            cols["tile"].append(np.full(n, t))
            for k in ("x", "y", "score"):
                cols[k].append(d[k].float().cpu().numpy())
            for k in ("radius", "m200m"):
                v = d.get(k)
                cols[k].append(np.full(n, np.nan) if v is None else v.float().cpu().numpy())
    return Detections(**{k: np.concatenate(v) for k, v in cols.items()})


def in_footprint(x: np.ndarray, y: np.ndarray, shape: tuple[int, int], edge: float) -> np.ndarray:
    """
    Whether pixel positions lie at least ``edge`` pixels inside a tile.

    Parameters
    ----------
    x, y : np.ndarray
        0-based pixel positions.
    shape : tuple[int, int]
        Tile shape ``(ny, nx)``.
    edge : float
        Margin in pixels.

    Returns
    -------
    inside : np.ndarray
        Boolean array.
    """
    ny, nx = shape
    return (x >= edge) & (x <= nx - 1 - edge) & (y >= edge) & (y <= ny - 1 - edge)


def to_sky(dets: Detections, tiles: PackedTiles) -> Detections:
    """
    Fill in the sky coordinates of detections.

    Parameters
    ----------
    dets : Detections
        Detections with tile indices and pixel positions.
    tiles : PackedTiles
        Tile geometry.

    Returns
    -------
    dets : Detections
        The same object with ``ra``/``dec`` set.
    """
    dets.ra, dets.dec = tiles.wcs.pix2world(dets.x, dets.y, dets.tile)
    return dets


def truth_in_footprint(
    tiles: PackedTiles, idx: np.ndarray, m_min: float, edge: float, pad: float = 4.0
) -> dict[str, np.ndarray]:
    """
    Unique catalogue clusters above a mass cut in (or just around) the footprint of some tiles.

    Parameters
    ----------
    tiles : PackedTiles
        Packed tiles with targets.
    idx : np.ndarray
        Tile indices defining the footprint.
    m_min : float
        Minimum M200m.
    edge : float
        Tile margin in pixels defining the footprint (as used for detections).
    pad : float, default=4.0
        Clusters up to this many pixels outside the footprint are also returned
        (flagged ``inner=False``) so boundary detections can still match them.

    Returns
    -------
    truth : dict[str, np.ndarray]
        ``cat_idx``, ``ra``, ``dec`` (from the tile projection), ``m200m``,
        ``z``, ``theta500c`` and ``inner`` (inside the strict footprint) per
        unique cluster.
    """
    tg = tiles.targets
    rows = np.concatenate([np.arange(tg.offsets[i], tg.offsets[i + 1]) for i in idx])
    tile = np.repeat(idx, np.diff(tg.offsets)[idx])
    x, y = tg.x[rows], tg.y[rows]
    keep = in_footprint(x, y, tiles.shape, edge - pad) & (tg.m200m[rows] >= m_min)
    inner = in_footprint(x, y, tiles.shape, edge)[keep]
    rows, tile = rows[keep], tile[keep]
    uniq, first, inverse = np.unique(tg.cat_idx[rows], return_index=True, return_inverse=True)
    any_inner = np.zeros(len(uniq), bool)
    np.logical_or.at(any_inner, inverse, inner)
    rows, tile = rows[first], tile[first]
    ra, dec = tiles.wcs.pix2world(tg.x[rows], tg.y[rows], tile)
    return {
        "cat_idx": uniq,
        "ra": ra,
        "dec": dec,
        "m200m": tg.m200m[rows],
        "z": tg.z[rows],
        "theta500c": tg.theta500c[rows],
        "inner": any_inner,
    }


@dataclass
class CatalogScores:
    """
    Detection catalogue matched to the truth.

    Attributes
    ----------
    det : Detections
        De-duplicated detections, sorted by decreasing score.
    det_match : np.ndarray
        Index into ``truth`` of each detection's match, or -1.
    det_real : np.ndarray
        Whether the detection lies on any halo above ``m_ignore`` (a looser purity).
    truth : dict[str, np.ndarray]
        Truth clusters (see :func:`truth_in_footprint`).
    truth_score : np.ndarray
        Score of the detection matched to each truth (0 if undetected).
        Completeness uses only truths with ``truth["inner"]``.
    """

    det: Detections
    det_match: np.ndarray
    det_real: np.ndarray
    truth: dict[str, np.ndarray]
    truth_score: np.ndarray

    def curves(self, thresholds: np.ndarray) -> dict[str, np.ndarray]:
        """
        Purity and completeness as a function of score threshold.

        Greedy matching and de-duplication both run in score order, so the
        catalogue at a threshold is a prefix of the full catalogue and one
        matching serves every threshold.

        Parameters
        ----------
        thresholds : np.ndarray
            Score thresholds.

        Returns
        -------
        curves : dict[str, np.ndarray]
            ``threshold``, ``n_det``, ``purity``, ``purity_ext``, ``completeness``, ``f1``.
        """
        s = self.det.score
        n_det = np.array([(s >= t).sum() for t in thresholds])
        n_tp = np.array([((s >= t) & (self.det_match >= 0)).sum() for t in thresholds])
        n_real = np.array([((s >= t) & self.det_real).sum() for t in thresholds])
        purity = n_tp / np.maximum(n_det, 1)
        inner = self.truth_score[self.truth["inner"]]
        comp = np.array([(inner >= t).mean() for t in thresholds])
        f1 = 2 * purity * comp / np.maximum(purity + comp, 1e-12)
        return {
            "threshold": np.asarray(thresholds),
            "n_det": n_det,
            "purity": purity,
            "purity_ext": n_real / np.maximum(n_det, 1),
            "completeness": comp,
            "f1": f1,
        }

    def best_threshold(self, thresholds: np.ndarray | None = None) -> float:
        """
        Threshold that maximises F1.

        Parameters
        ----------
        thresholds : np.ndarray, optional
            Candidate thresholds (defaults to a fine grid).

        Returns
        -------
        t : float
            Best threshold.
        """
        thresholds = np.linspace(0.02, 0.99, 98) if thresholds is None else thresholds
        c = self.curves(thresholds)
        return float(c["threshold"][np.argmax(c["f1"])])


def score_catalog(
    dets: Detections,
    tiles: PackedTiles,
    idx: np.ndarray,
    cfg: MaskConfig,
    edge: float = 2.0,
    match_arcmin: float = 2.0,
    dedupe_arcmin: float = 2.0,
    mass_cut: float | None = None,
) -> CatalogScores:
    """
    De-duplicate detections across overlapping tiles and match them to the truth.

    Parameters
    ----------
    dets : Detections
        Raw detections over the tiles ``idx``.
    tiles : PackedTiles
        Packed tiles with targets.
    idx : np.ndarray
        Tile indices that were searched.
    cfg : MaskConfig
        Provides the positive (``m_cut``) and "real but sub-threshold"
        (``m_ignore``) mass cuts.
    edge : float, default=2.0
        Ignore detections and truths within this many pixels of a tile edge.
    match_arcmin : float, default=2.0
        Matching radius.
    dedupe_arcmin : float, default=2.0
        Detections closer than this are merged (highest score kept).
    mass_cut : float, optional
        Drop detections whose predicted M200m is below this (detectors with
        a mass head only; NaN masses are kept).

    Returns
    -------
    scores : CatalogScores
        Matched catalogue.
    """
    dets = to_sky(dets.select(in_footprint(dets.x, dets.y, tiles.shape, edge)), tiles)
    if mass_cut is not None:
        dets = dets.select(~(dets.m200m < mass_cut))
    dets = dets.select(dedupe(dets.ra, dets.dec, dets.score, dedupe_arcmin))
    truth = truth_in_footprint(tiles, idx, cfg.m_cut, edge)
    match = one_to_one_match(dets.ra, dets.dec, dets.score, truth["ra"], truth["dec"], match_arcmin)
    truth_score = np.zeros(len(truth["ra"]))
    truth_score[match[match >= 0]] = dets.score[match >= 0]
    ext = truth_in_footprint(tiles, idx, cfg.m_ignore, edge)
    near, _ = match_catalogs(dets.ra, dets.dec, ext["ra"], ext["dec"], match_arcmin)
    return CatalogScores(dets, match, (match >= 0) | (near >= 0), truth, truth_score)


def in_hpx(ra: np.ndarray, dec: np.ndarray, footprint: np.ndarray, nside: int) -> np.ndarray:
    """
    Whether positions fall in a HEALPix footprint.

    Parameters
    ----------
    ra, dec : np.ndarray
        Positions [deg].
    footprint : np.ndarray
        Sorted RING pixel indices of the footprint.
    nside : int
        HEALPix resolution of ``footprint``.

    Returns
    -------
    inside : np.ndarray
        Boolean per position.
    """
    pix = hp.ang2pix(nside, ra, dec, lonlat=True)
    j = np.clip(np.searchsorted(footprint, pix), 0, len(footprint) - 1)
    return footprint[j] == pix


def score_sky(
    ra: np.ndarray,
    dec: np.ndarray,
    score: np.ndarray,
    cat: Catalog,
    footprint: np.ndarray,
    nside: int,
    cfg: MaskConfig,
    match_arcmin: float = 2.0,
) -> dict[str, np.ndarray]:
    """
    Score a blind sky catalogue inside a footprint (any detector: network or matched filter).

    Detections and truths are both restricted to the footprint. Truths are catalogue halos with
    M200m >= ``cfg.m_cut``; a detection is "real" if it lies within ``match_arcmin`` of any halo
    with M200m >= ``cfg.m_ignore`` (inside the footprint or not).

    Parameters
    ----------
    ra, dec, score : np.ndarray
        De-duplicated detections [deg] and their scores.
    cat : Catalog
        Halo catalogue.
    footprint : np.ndarray
        Sorted HEALPix RING pixels of the searched area.
    nside : int
        Resolution of ``footprint``.
    cfg : MaskConfig
        Mass cuts.
    match_arcmin : float, default=2.0
        Matching radius.

    Returns
    -------
    res : dict[str, np.ndarray]
        ``det_ra``, ``det_dec``, ``det_score``, ``det_match`` (truth index or -1), ``det_real``,
        ``truth_*`` columns (``cat_idx``, ``ra``, ``dec``, ``m200m``, ``m500c``, ``z``,
        ``theta500c``, ``inner`` = all True) and ``truth_score`` (0 if missed).
    """
    keep = in_hpx(ra, dec, footprint, nside)
    ra, dec, score = ra[keep], dec[keep], score[keep]
    ext = np.where(cat.m200m >= cfg.m_ignore)[0]
    t = ext[(cat.m200m[ext] >= cfg.m_cut) & in_hpx(cat.ra[ext], cat.dec[ext], footprint, nside)]
    match = one_to_one_match(ra, dec, score, cat.ra[t], cat.dec[t], match_arcmin)
    t_score = np.zeros(len(t))
    t_score[match[match >= 0]] = score[match >= 0]
    near, _ = match_catalogs(ra, dec, cat.ra[ext], cat.dec[ext], match_arcmin)
    truth = {k: getattr(cat, k)[t] for k in ("ra", "dec", "m200m", "m500c", "z", "theta500c")}
    return {
        "det_ra": ra,
        "det_dec": dec,
        "det_score": score,
        "det_match": match,
        "det_real": (match >= 0) | (near >= 0),
        "truth_score": t_score,
        "truth_cat_idx": t,
        "truth_inner": np.ones(len(t), bool),
        **{f"truth_{k}": v for k, v in truth.items()},
    }


def sky_curves(res: dict[str, np.ndarray], thresholds: np.ndarray) -> dict[str, list[float]]:
    """
    Purity, completeness and F1 versus threshold for :func:`score_sky` output.

    Parameters
    ----------
    res : dict[str, np.ndarray]
        Output of :func:`score_sky`.
    thresholds : np.ndarray
        Score thresholds.

    Returns
    -------
    curves : dict[str, list[float]]
        ``threshold``, ``n_det``, ``purity``, ``purity_ext``, ``completeness``, ``f1``.
    """
    s, m, r, ts = res["det_score"], res["det_match"] >= 0, res["det_real"], res["truth_score"]
    out: dict[str, list[float]] = {
        k: [] for k in ("threshold", "n_det", "purity", "purity_ext", "completeness", "f1")
    }
    for t in thresholds:
        sel = s >= t
        n = max(int(sel.sum()), 1)
        p, c = (sel & m).sum() / n, (ts >= t).mean()
        vals = (t, sel.sum(), p, (sel & r).sum() / n, c, 2 * p * c / max(p + c, 1e-9))
        for k, v in zip(out, vals, strict=True):
            out[k].append(float(v))
    return out
