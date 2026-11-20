"""Cut large overlapping TAN patches covering the ACT-footprint test sky, for blind evaluation.

The test footprint is the union of the strict footprints of the ``split_act.npz`` test tiles
(the area nemo searches, see ``make_nemo_config.py``); it is saved as sorted HEALPix pixels
(``footprint_hpx<nside>.npy``). Patches of ``--size`` degrees on a Dec/RA grid with ``--overlap``
are kept if they touch the footprint, and cut from each map with the same code as training
tiles (``mlclusters.stamps.make_tiles``), giving a packed directory usable by ``eval_sky.py``.
"""

import argparse as argp
from pathlib import Path

import healpy as hp
import numpy as np

from mlclusters.data import PackedTiles
from mlclusters.geometry import footprint_hpx
from mlclusters.stamps import make_tiles

OBS = Path("/mnt/welch/USERS/jorlo/ml-clusters/maps/observed")


def patch_centres(footprint: np.ndarray, nside: int, size: float, overlap: float) -> np.ndarray:
    """
    Centres of a Dec/RA grid of patches that touch the footprint.

    Parameters
    ----------
    footprint : np.ndarray
        Footprint HEALPix pixels.
    nside : int
        Their resolution.
    size, overlap : float
        Patch side and overlap [deg].

    Returns
    -------
    centres : np.ndarray
        ``(n, 2)`` RA, Dec [deg].
    """
    low = 64
    ra_f, dec_f = hp.pix2ang(nside, footprint, lonlat=True)
    lowmap = np.zeros(hp.nside2npix(low), bool)
    lowmap[hp.ang2pix(low, ra_f, dec_f, lonlat=True)] = True
    step = size - overlap
    out = []
    for dec in np.arange(dec_f.min() - size / 2 + step / 2, dec_f.max() + size / 2, step):
        n_ra = int(np.ceil(360 * np.cos(np.radians(dec)) / step))
        for ra in (np.arange(n_ra) + 0.5) * 360 / n_ra:
            disc = hp.query_disc(low, hp.ang2vec(ra, dec, lonlat=True), np.radians(size / 2))
            if lowmap[disc].any():
                out.append((ra, dec))
    return np.array(out)


def _make_parser() -> argp.ArgumentParser:
    """
    Build the CLI parser.

    Returns
    -------
    parser : argp.ArgumentParser
        Parser.
    """
    p = argp.ArgumentParser(description=__doc__, formatter_class=argp.RawDescriptionHelpFormatter)
    p.add_argument("tiles", type=Path, help="1 deg packed tiles with split_act.npz.")
    p.add_argument("out_dir", type=Path, help="Output packed directory.")
    p.add_argument(
        "--maps",
        nargs=3,
        default=[str(OBS / f"websky_f{b}_beam_noise.fits") for b in ("090", "150", "220")],
    )
    p.add_argument("--size", type=float, default=4.0, help="Patch side [deg].")
    p.add_argument("--overlap", type=float, default=0.5, help="Patch overlap [deg].")
    p.add_argument("--edge", type=float, default=2.0, help="1 deg tile margin for the footprint.")
    p.add_argument("--nside", type=int, default=4096)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--debug", action="store_true", help="First 10 patches only.")
    return p


def main() -> None:
    """Write the footprint and the patches."""
    args = _make_parser().parse_args()
    tiles = PackedTiles(args.tiles)
    idx = np.where(np.load(args.tiles / "split_act.npz")["split"] == 2)[0]
    hit = footprint_hpx(tiles.wcs[idx], tiles.shape, args.edge, args.nside, tiles.pix_arcmin)
    footprint = np.where(hit)[0]
    area = len(footprint) * hp.nside2pixarea(args.nside, degrees=True)
    centres = patch_centres(footprint, args.nside, args.size, args.overlap)
    if args.debug:
        centres = centres[:10]
    print(f"footprint {area:.0f} deg^2; {len(centres)} patches of {args.size} deg", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / f"footprint_hpx{args.nside}.npy", footprint)
    make_tiles(
        args.maps,
        centres[:, 0],
        centres[:, 1],
        args.out_dir,
        half_width_arcmin=args.size * 30,
        n_workers=args.workers,
        chunk=8,
    )
    np.savez(args.out_dir / "split.npz", split=np.full(len(centres), 2))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
