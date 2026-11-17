"""Read a cluster catalogue (WebSky .pksc or nemo .fits), add sizes and cache it as .npz."""

import argparse as argp
from pathlib import Path

import numpy as np

from mlclusters.catalogs import read_fits_catalog, read_websky_halos


def _make_parser() -> argp.ArgumentParser:
    """
    Build the CLI parser.

    Returns
    -------
    parser : argp.ArgumentParser
        Parser.
    """
    parser = argp.ArgumentParser(description=__doc__)
    parser.add_argument("cat_path", type=Path, help="halos.pksc or a nemo FITS catalogue.")
    parser.add_argument("out_path", type=Path, help="Output .npz path.")
    parser.add_argument("--m_min", type=float, default=1e14, help="Minimum M200m (WebSky).")
    parser.add_argument("--z_min", type=float, default=0.05, help="Minimum redshift (WebSky).")
    parser.add_argument("--debug", action="store_true", help="Read only the 10^5 most massive.")
    return parser


def main() -> None:
    """Build and save the catalogue."""
    args = _make_parser().parse_args()
    if args.cat_path.suffix == ".pksc":
        cat = read_websky_halos(
            args.cat_path,
            m_min=args.m_min,
            z_min=args.z_min,
            max_rows=100_000 if args.debug else None,
        )
    else:
        cat = read_fits_catalog(args.cat_path)
    cat.save(args.out_path)
    t = cat.theta500c[np.isfinite(cat.theta500c)]
    pct = np.percentile(t, [5, 50, 95])
    print(f"{len(cat)} clusters; theta500c (5, 50, 95)th percentiles = {pct} arcmin")


if __name__ == "__main__":
    main()
