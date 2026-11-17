"""Evaluate a detector: build sky catalogues on val/test tiles and measure purity/completeness."""

import argparse as argp
import json
from pathlib import Path

import numpy as np
import torch

from mlclusters.data import Normalizer, PackedTiles, TileDataset
from mlclusters.detect import CatalogScores, predict, score_catalog
from mlclusters.legacy import frcnn_detector, load_legacy_frcnn
from mlclusters.masks import MaskConfig
from mlclusters.training import heat_detector, load_checkpoint

THRESHOLDS = np.round(np.linspace(0.02, 0.99, 98), 2)


def _make_parser() -> argp.ArgumentParser:
    """
    Build the CLI parser.

    Returns
    -------
    parser : argp.ArgumentParser
        Parser.
    """
    p = argp.ArgumentParser(description=__doc__)
    p.add_argument("tiles", type=Path, help="Packed tile directory (with targets/split).")
    p.add_argument("model", type=Path, help="ClusterNet checkpoint, or legacy .pth with --legacy.")
    p.add_argument("out", type=Path, help="Output prefix for <out>.json and <out>.npz.")
    p.add_argument("--legacy", action="store_true", help="Model is a legacy Faster R-CNN.")
    p.add_argument("--match_arcmin", type=float, default=2.0, help="Matching radius.")
    p.add_argument("--edge", type=float, default=2.0, help="Tile edge margin [pix].")
    p.add_argument("--batch_size", "-b", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--split", default="split.npz", help="Split file in the tile directory.")
    p.add_argument("--debug", action="store_true", help="Only use 300 tiles per split.")
    return p


def _scores(
    split_name: str,
    idx: np.ndarray,
    args: argp.Namespace,
    tiles: PackedTiles,
    norm: Normalizer,
    cfg: MaskConfig,
    detector: object,
    device: torch.device,
) -> CatalogScores:
    """
    Run the detector over one split and score the resulting catalogue.

    Parameters
    ----------
    split_name : str
        Name for logging.
    idx : np.ndarray
        Tile indices.
    args : argp.Namespace
        CLI arguments.
    tiles : PackedTiles
        Packed tiles.
    norm : Normalizer
        Input normalisation.
    cfg : MaskConfig
        Mass cuts.
    detector : Detector
        Detector callable.
    device : torch.device
        Device.

    Returns
    -------
    scores : CatalogScores
        Matched catalogue.
    """
    ds = TileDataset(tiles, idx, norm, cfg, with_targets=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
    dets = predict(detector, dl, device)
    print(f"{split_name}: {len(idx)} tiles, {len(dets.score)} raw detections", flush=True)
    return score_catalog(dets, tiles, idx, cfg, args.edge, args.match_arcmin, args.match_arcmin)


def main() -> None:
    """Evaluate and save curves and per-object tables."""
    args = _make_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tiles = PackedTiles(args.tiles)
    split = np.load(args.tiles / args.split)["split"]
    if args.legacy:
        model, cfg = load_legacy_frcnn(args.model, device), MaskConfig()
        norm, detector = Normalizer.identity(tiles.images.shape[1]), frcnn_detector(model)
        args.batch_size = min(args.batch_size, 16)
    else:
        model, cfg, norm, _ = load_checkpoint(args.model, device)
        model = model.to(memory_format=torch.channels_last)
        detector = heat_detector(model, threshold=float(THRESHOLDS[0]))

    results, tables = {}, {}
    for name, code in (("val", 1), ("test", 2)):
        idx = np.where(split == code)[0]
        idx = idx[:300] if args.debug else idx
        sc = _scores(name, idx, args, tiles, norm, cfg, detector, device)
        results[name] = {k: v.tolist() for k, v in sc.curves(THRESHOLDS).items()}
        tables[name] = sc
    thr = tables["val"].best_threshold(THRESHOLDS)
    test = tables["test"]
    at_thr = {k: float(v[0]) for k, v in test.curves(np.array([thr])).items()}
    results["chosen"] = {"threshold_from_val": thr, "test": at_thr}
    print(json.dumps(results["chosen"], indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(results, indent=2))
    det = test.det
    np.savez(
        args.out.with_suffix(".npz"),
        threshold=thr,
        det_ra=det.ra,
        det_dec=det.dec,
        det_score=det.score,
        det_radius_pix=det.radius,
        det_match=test.det_match,
        det_real=test.det_real,
        truth_score=test.truth_score,
        **{f"truth_{k}": v for k, v in test.truth.items()},
    )


if __name__ == "__main__":
    main()
