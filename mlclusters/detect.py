"""Run a detector over tiles, build a sky catalogue and score it against the truth."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

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
    ra, dec : np.ndarray
        Sky position in degrees (filled by :func:`to_sky`).
    """

    tile: np.ndarray
    x: np.ndarray
    y: np.ndarray
    score: np.ndarray
    radius: np.ndarray
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
    cols: dict[str, list[np.ndarray]] = {k: [] for k in ("tile", "x", "y", "score", "radius")}
    for batch in loader:
        dets = detector(batch["image"].to(device, non_blocking=True))
        for t, d in zip(batch["index"].numpy(), dets, strict=True):
            n = len(d["score"])
            cols["tile"].append(np.full(n, t))
            for k in ("x", "y", "score"):
                cols[k].append(d[k].float().cpu().numpy())
            r = d.get("radius")
            cols["radius"].append(np.full(n, np.nan) if r is None else r.float().cpu().numpy())
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
        thresholds = np.linspace(0.05, 0.95, 91) if thresholds is None else thresholds
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

    Returns
    -------
    scores : CatalogScores
        Matched catalogue.
    """
    dets = to_sky(dets.select(in_footprint(dets.x, dets.y, tiles.shape, edge)), tiles)
    dets = dets.select(dedupe(dets.ra, dets.dec, dets.score, dedupe_arcmin))
    truth = truth_in_footprint(tiles, idx, cfg.m_cut, edge)
    match = one_to_one_match(dets.ra, dets.dec, dets.score, truth["ra"], truth["dec"], match_arcmin)
    truth_score = np.zeros(len(truth["ra"]))
    truth_score[match[match >= 0]] = dets.score[match >= 0]
    ext = truth_in_footprint(tiles, idx, cfg.m_ignore, edge)
    near, _ = match_catalogs(dets.ra, dets.dec, ext["ra"], ext["dec"], match_arcmin)
    return CatalogScores(dets, match, (match >= 0) | (near >= 0), truth, truth_score)
