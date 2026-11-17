"""Cut multi-frequency tiles around catalogue clusters (or random points) from full-sky maps."""

import argparse as argp
from pathlib import Path

import numpy as np

from mlclusters.catalogs import load_catalog
from mlclusters.data import TILES
from mlclusters.stamps import make_tiles, tile_centres


def _make_parser() -> argp.ArgumentParser:
    """
    Build the CLI parser.

    Returns
    -------
    parser : argp.ArgumentParser
        Parser.
    """
    p = argp.ArgumentParser(description=__doc__)
    p.add_argument("catalog", type=Path, help="Catalogue (.npz from build_catalog.py, or .fits).")
    p.add_argument(
        "--centres_from", type=Path, help="Re-use tile centres of a packed directory instead."
    )
    p.add_argument("out_dir", type=Path, help="Output packed-tile directory.")
    p.add_argument("maps", nargs="+", help="Full-sky maps, one per frequency, in channel order.")
    p.add_argument("--m_min", type=float, default=2e14, help="Centre tiles on M200m above this.")
    p.add_argument("--offset", type=float, default=5.0, help="Max centre offset [arcmin].")
    p.add_argument("--half_width", type=float, default=30.0, help="Tile half-width [arcmin].")
    p.add_argument("--res", type=float, default=0.5, help="Pixel size [arcmin].")
    p.add_argument("--n_random", type=int, default=0, help="Also add this many random tiles.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_workers", "-n", type=int, default=4, help="Worker processes.")
    p.add_argument("--debug", action="store_true", help="Only make 50 tiles.")
    return p


def main() -> None:
    """Make the tiles."""
    args = _make_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    if args.centres_from is not None:
        with np.load(args.centres_from / TILES) as t:
            ra, dec = t["crval1"], t["crval2"]
    else:
        cat = load_catalog(args.catalog)
        sel = ~(cat.m200m < args.m_min)  # keep NaN masses (e.g. observed catalogues)
        ra, dec = tile_centres(cat.ra[sel], cat.dec[sel], args.offset, rng)
    if args.n_random:
        ra = np.concatenate([ra, rng.uniform(0, 360, args.n_random)])
        dec = np.concatenate([dec, np.degrees(np.arcsin(rng.uniform(-1, 1, args.n_random)))])
    if args.debug:
        ra, dec = ra[:50], dec[:50]
    print(f"Making {len(ra)} tiles")
    make_tiles(
        args.maps, ra, dec, args.out_dir, args.half_width, args.res, n_workers=args.n_workers
    )


if __name__ == "__main__":
    main()
