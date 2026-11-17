"""Make a full-sky WebSky temperature map for one band (CMB+kSZ+tSZ+CIB+radio [+noise])."""

import argparse as argp
from pathlib import Path

from pixell import enmap

from mlclusters.websky import BANDS, make_websky_map


def _make_parser() -> argp.ArgumentParser:
    """
    Build the CLI parser.

    Returns
    -------
    parser : argp.ArgumentParser
        Parser.
    """
    p = argp.ArgumentParser(description=__doc__)
    p.add_argument("band", choices=list(BANDS), help="Frequency band.")
    p.add_argument("out", type=Path, help="Output FITS map.")
    p.add_argument("--root", type=Path, default=Path("/mnt/welch/USERS/cwhitaker/maps/websky"))
    p.add_argument("--noise", type=float, default=None, help="White noise [uK-arcmin].")
    p.add_argument("--seed", type=int, default=None, help="Noise seed.")
    p.add_argument("--debug", action="store_true", help="Make a 4 arcmin resolution map.")
    return p


def main() -> None:
    """Make and write the map."""
    args = _make_parser().parse_args()
    res = 4.0 if args.debug else 0.5
    omap = make_websky_map(args.band, args.root, res, args.noise, args.seed)
    enmap.write_map(str(args.out), omap)


if __name__ == "__main__":
    main()
