"""Restrict a spatial split to tiles fully inside the ACT footprint, and record per-tile depth.

Samples each band's ivar on an 11x11 grid over every tile. Tiles with any unobserved sample in any
band become -1 (excluded); the others keep their label from ``split.npz``. Writes
``split_act.npz`` (``split``, ``depth`` [uK arcmin, median per tile and band]) into each packed
directory given (they must share tile centres, e.g. clean / beam / beam+noise cuts).
"""

import argparse as argp
from pathlib import Path

import numpy as np
from pixell import enmap

from mlclusters.data import PackedTiles
from mlclusters.geometry import TileWCS


def tile_depths(wcs: TileWCS, shape: tuple[int, int], ivar: enmap.ndmap, n: int = 11) -> np.ndarray:
    """
    Depth sampled on an ``n x n`` grid over each tile.

    Parameters
    ----------
    wcs : TileWCS
        Tile geometry.
    shape : tuple[int, int]
        Tile shape ``(ny, nx)``.
    ivar : enmap.ndmap
        Full-sky ivar [1/uK^2 per pixel].
    n : int, default=11
        Grid points per side.

    Returns
    -------
    depth : np.ndarray
        ``(n_tiles, n*n)`` depth [uK arcmin]; inf where unobserved.
    """
    ny, nx = shape
    pix_area = np.abs(ivar.wcs.wcs.cdelt[0] * ivar.wcs.wcs.cdelt[1]) * 3600.0  # arcmin^2 at equator
    out = []
    for py in np.linspace(0, ny - 1, n):
        for px in np.linspace(0, nx - 1, n):
            ra, dec = wcs.pix2world(np.full(len(wcs), px), np.full(len(wcs), py))
            iy, ix = np.round(ivar.sky2pix(np.deg2rad([dec, ra]))).astype(int)
            iy = np.clip(iy, 0, ivar.shape[-2] - 1)
            ix %= ivar.shape[-1]
            iv = np.asarray(ivar[iy, ix], dtype=np.float64)
            area = pix_area * np.cos(np.deg2rad(dec))
            with np.errstate(divide="ignore"):
                out.append(np.where(iv > 0, np.sqrt(area / iv), np.inf))
    return np.stack(out, axis=1)


def _make_parser() -> argp.ArgumentParser:
    """
    Build the CLI parser.

    Returns
    -------
    parser : argp.ArgumentParser
        Parser.
    """
    p = argp.ArgumentParser(description=__doc__)
    p.add_argument("tiles", type=Path, nargs="+", help="Packed directories (same tile centres).")
    p.add_argument("--ivar", nargs="+", required=True, help="Ivar maps, one per band.")
    p.add_argument("--out", default="split_act.npz", help="Output split file name.")
    p.add_argument("--debug", action="store_true", help="First 1000 tiles; print only.")
    return p


def main() -> None:
    """Compute and save the footprint split."""
    args = _make_parser().parse_args()
    tiles = PackedTiles(args.tiles[0])
    wcs = tiles.wcs[np.arange(1000)] if args.debug else tiles.wcs
    depth = []
    for path in args.ivar:
        d = tile_depths(wcs, tiles.shape, enmap.read_map(path))
        inside = np.isfinite(d).all(axis=1)
        depth.append(np.where(inside, np.median(d, axis=1), np.inf))
        print(f"{Path(path).name}: {inside.mean():.1%} of tiles fully observed", flush=True)
    depth = np.stack(depth, axis=1)
    inside = np.isfinite(depth).all(axis=1)
    print("median depth per band [uK arcmin]:", np.round(np.median(depth[inside], axis=0), 1))
    if args.debug:
        return
    for d in args.tiles:
        split = np.where(inside, np.load(d / "split.npz")["split"], -1)
        np.savez(d / args.out, split=split, depth=depth)
        print(f"{d}: (excluded, train, val, test) = {np.bincount(split + 1, minlength=4)}")


if __name__ == "__main__":
    main()
