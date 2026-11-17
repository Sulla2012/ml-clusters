"""Assign catalogue clusters (with mass-scaled sizes) to packed tiles and make a spatial split."""

import argparse as argp
from pathlib import Path

import numpy as np

from mlclusters.catalogs import load_catalog
from mlclusters.data import TARGETS, PackedTiles
from mlclusters.geometry import tile_split
from mlclusters.masks import compute_tile_targets

SPLIT = "split.npz"


def _make_parser() -> argp.ArgumentParser:
    """
    Build the CLI parser.

    Returns
    -------
    parser : argp.ArgumentParser
        Parser.
    """
    parser = argp.ArgumentParser(description=__doc__)
    parser.add_argument("tiles", type=Path, help="Packed tile directory.")
    parser.add_argument("catalog", type=Path, help="Catalogue (.npz from build_catalog.py).")
    parser.add_argument("--margin", type=float, default=7.0, help="Off-tile margin [arcmin].")
    parser.add_argument("--nside", type=int, default=4, help="HEALPix nside of split regions.")
    parser.add_argument(
        "--fractions", type=float, nargs=3, default=(0.8, 0.1, 0.1), help="Train/val/test."
    )
    parser.add_argument("--seed", type=int, default=0, help="Split seed.")
    parser.add_argument("--debug", action="store_true", help="Print a summary of tile 0 too.")
    return parser


def main() -> None:
    """Compute and save targets and split."""
    args = _make_parser().parse_args()
    tiles = PackedTiles(args.tiles)
    cat = load_catalog(args.catalog)
    targets = compute_tile_targets(tiles.wcs, tiles.shape, cat, args.margin)
    targets.save(args.tiles / TARGETS)
    split = tile_split(tiles.wcs, tiles.shape, tuple(args.fractions), args.nside, args.seed)
    np.savez(args.tiles / SPLIT, split=split, nside=args.nside, seed=args.seed)
    counts = np.diff(targets.offsets)
    print(f"{len(tiles)} tiles, {len(targets.x)} cluster rows, median {np.median(counts)}/tile")
    print("split counts (boundary, train, val, test):", np.bincount(split + 1, minlength=4))
    if args.debug:
        print("tile 0:", {k: np.round(v, 2) for k, v in targets.tile(0).items()})


if __name__ == "__main__":
    main()
