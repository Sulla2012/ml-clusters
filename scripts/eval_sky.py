"""Blind evaluation on the test sky: ClusterNet on large patches, or a nemo catalogue.

Both are scored by :func:`mlclusters.detect.score_sky` inside the same HEALPix footprint (written
by ``make_sky_patches.py``) against every catalogue halo there, so the two are directly
comparable. ClusterNet detections within ``--edge`` pixels of a patch border are dropped and the
rest de-duplicated across overlapping patches at 2'. Writes ``<out>.json`` (curves, plus the
operating point at ``--threshold``) and ``<out>.npz`` (the ``evaluate.py`` format, read by
``compare_act.py``).

``--rescore`` re-scores the detections of an earlier ``<out>.npz`` (e.g. on a smaller footprint)
without re-running anything. ``--good_tiles`` restricts the footprint to the nemo tiles whose
filters are sane (median fixed_y_c of their candidates below ``--max_yc``; tiles with exploded
filters give y_c ~ 1e3-1e9).
"""

import argparse as argp
import json
from pathlib import Path

import healpy as hp
import numpy as np
import torch
import yaml
from astropy.table import Table

from mlclusters.catalogs import Catalog
from mlclusters.data import PackedTiles, TileDataset
from mlclusters.detect import in_footprint, predict, score_sky, sky_curves, to_sky
from mlclusters.geometry import dedupe, erode_hpx, footprint_hpx
from mlclusters.masks import MaskConfig
from mlclusters.training import heat_detector, load_checkpoint

GRIDS = {
    "model": np.round(np.linspace(0.05, 0.99, 95), 2),
    "nemo": np.round(np.arange(3.5, 15.01, 0.1), 2),
}


def clusternet_catalog(
    patches: PackedTiles, ckpt: Path, edge: float, batch: int, debug: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a ClusterNet checkpoint over all patches and build a de-duplicated sky catalogue.

    Parameters
    ----------
    patches : PackedTiles
        Large patches.
    ckpt : Path
        Checkpoint.
    edge : float
        Patch border margin [pix].
    batch : int
        Batch size.
    debug : bool
        First 4 patches only.

    Returns
    -------
    ra, dec, score : np.ndarray
        Catalogue.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, norm, _ = load_checkpoint(ckpt, device)
    model.eval()
    idx = np.arange(4 if debug else len(patches))
    ds = TileDataset(patches, idx, norm, cfg, with_targets=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=4)
    dets = predict(heat_detector(model, threshold=0.05, max_det=20000), dl, device)
    dets = to_sky(dets.select(in_footprint(dets.x, dets.y, patches.shape, edge)), patches)
    keep = dedupe(dets.ra, dets.dec, dets.score, 2.0)
    print(f"{len(idx)} patches: {len(dets.score)} detections, {len(keep)} after de-dup", flush=True)
    return dets.ra[keep], dets.dec[keep], dets.score[keep]


def good_tile_footprint(
    footprint: np.ndarray, nside: int, nemo_cat: Path, nemo_cfg: Path, max_yc: float
) -> np.ndarray:
    """
    Restrict a footprint to the cores of nemo tiles whose filters look sane.

    Parameters
    ----------
    footprint : np.ndarray
        Sorted HEALPix RING pixels.
    nside : int
        Resolution of ``footprint``.
    nemo_cat : Path
        nemo ``*_optimalCatalog.fits``.
    nemo_cfg : Path
        nemo config with explicit ``tileDefinitions`` (``RADecSection`` boxes).
    max_yc : float
        Keep tiles whose candidates have median ``fixed_y_c`` below this [1e-4].

    Returns
    -------
    footprint : np.ndarray
        Sorted pixels inside the good tiles.
    """
    tab = Table.read(nemo_cat)
    names = np.char.lower(np.asarray(tab["tileName"]).astype(str))
    y = np.asarray(tab["fixed_y_c"], float)
    good = {k for k in np.unique(names) if np.median(y[names == k]) < max_yc}
    ra, dec = hp.pix2ang(nside, footprint, lonlat=True)
    keep = np.zeros(len(footprint), bool)
    for t in yaml.safe_load(nemo_cfg.read_text())["tileDefinitions"]:
        if t["tileName"] in good:
            r0, r1, d0, d1 = t["RADecSection"]
            keep |= (ra >= r0) & (ra < r1) & (dec >= d0) & (dec < d1)
    print(f"{len(good)} good nemo tiles: {keep.mean():.1%} of the footprint", flush=True)
    return footprint[keep]


def _make_parser() -> argp.ArgumentParser:
    """
    Build the CLI parser.

    Returns
    -------
    parser : argp.ArgumentParser
        Parser.
    """
    p = argp.ArgumentParser(description=__doc__, formatter_class=argp.RawDescriptionHelpFormatter)
    p.add_argument("patches", type=Path, help="Directory from make_sky_patches.py.")
    p.add_argument("out", type=Path, help="Output prefix.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", type=Path, help="ClusterNet checkpoint.")
    src.add_argument("--nemo", type=Path, help="nemo *_optimalCatalog.fits (score = fixed_SNR).")
    src.add_argument("--rescore", type=Path, help="Detections from an earlier <out>.npz.")
    p.add_argument(
        "--footprint", type=Path, help="footprint_hpx<nside>*.npy, intersected with the patches'."
    )
    p.add_argument("--erode_arcmin", type=float, default=0.0, help="Shrink the footprint edge.")
    p.add_argument("--good_tiles", type=Path, nargs=2, metavar=("NEMO_CAT", "NEMO_CFG"))
    p.add_argument("--max_yc", type=float, default=3.0, help="Good-tile median y_c limit.")
    p.add_argument("--catalog", type=Path, required=True, help="Halo catalogue.")
    p.add_argument("--threshold", type=float, help="Operating point (default 0.5 / S/N 5).")
    p.add_argument("--edge", type=float, default=20.0, help="Patch border margin [pix].")
    p.add_argument("--match_arcmin", type=float, default=2.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--debug", action="store_true", help="Four patches (model only).")
    return p


def main() -> None:
    """Build the catalogue, score it and save."""
    args = _make_parser().parse_args()
    fp_file = next(args.patches.glob("footprint_hpx*.npy"))
    nside = int(fp_file.stem.removeprefix("footprint_hpx"))
    footprint = np.load(fp_file)
    if args.footprint:
        assert int(args.footprint.stem.removeprefix("footprint_hpx").split("_")[0]) == nside
        footprint = np.intersect1d(footprint, np.load(args.footprint))
    patches = PackedTiles(args.patches)
    if args.model:
        ra, dec, score = clusternet_catalog(
            patches, args.model, args.edge, args.batch_size, args.debug
        )
        kind, thr = "model", 0.5 if args.threshold is None else args.threshold
    else:
        if args.nemo:
            tab = Table.read(args.nemo)
            ra, dec, score = (np.asarray(tab[k], float) for k in ("RADeg", "decDeg", "fixed_SNR"))
        else:
            d = np.load(args.rescore)
            ra, dec, score = d["det_ra"], d["det_dec"], d["det_score"]
        kind = "nemo" if score.max() > 1 else "model"
        thr = {"nemo": 5.0, "model": 0.5}[kind] if args.threshold is None else args.threshold
    if args.erode_arcmin > 0:
        n0 = len(footprint)
        footprint = erode_hpx(footprint, nside, args.erode_arcmin)
        print(f"eroded by {args.erode_arcmin}': {len(footprint) / n0:.1%} of the area kept")
    if args.good_tiles:
        footprint = good_tile_footprint(footprint, nside, *args.good_tiles, args.max_yc)
    if args.debug:  # only the sky the debug patches cover
        cover = footprint_hpx(patches.wcs[np.arange(4)], patches.shape, args.edge, nside)
        footprint = np.intersect1d(footprint, np.where(cover)[0])
    cat = Catalog.load(args.catalog)
    res = score_sky(ra, dec, score, cat, footprint, nside, MaskConfig(), args.match_arcmin)
    at_thr = sky_curves(res, np.array([thr]))
    out = {
        "test": sky_curves(res, GRIDS[kind]),
        "chosen": {"threshold_from_val": thr, "test": {k: v[0] for k, v in at_thr.items()}},
        "area_deg2": len(footprint) * hp.nside2pixarea(nside, degrees=True),
    }
    print(json.dumps(out["chosen"], indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(out, indent=2))
    np.savez(args.out.with_suffix(".npz"), threshold=thr, **res)
    print(f"wrote {args.out}.json/.npz")


if __name__ == "__main__":
    main()
