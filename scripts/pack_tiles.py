"""Pack directories of FITS tiles into a single memory-mapped array for fast training."""

import argparse as argp
from pathlib import Path

from mlclusters.data import pack_fits_tiles


def _make_parser() -> argp.ArgumentParser:
    """
    Build the CLI parser.

    Returns
    -------
    parser : argp.ArgumentParser
        Parser.
    """
    parser = argp.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Output directory for the packed tiles.")
    parser.add_argument("tile_dirs", type=Path, nargs="+", help="Directories of FITS tiles.")
    parser.add_argument("--n_workers", "-n", type=int, default=8, help="Reader processes.")
    parser.add_argument("--debug", action="store_true", help="Only pack 1000 tiles.")
    return parser


def main() -> None:
    """Pack tiles."""
    args = _make_parser().parse_args()
    paths = sorted((p for d in args.tile_dirs for p in d.glob("*.fits")), key=lambda p: int(p.stem))
    if args.debug:
        paths = paths[:1000]
    print(f"Packing {len(paths)} tiles into {args.out_dir}")
    pack_fits_tiles(paths, args.out_dir, n_workers=args.n_workers)


if __name__ == "__main__":
    main()
