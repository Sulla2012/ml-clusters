"""Cut multi-frequency gnomonic tiles out of full-sky maps."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
from pixell import enmap, reproject, utils

from .data import IMAGES, TILES
from .geometry import TileWCS

_SHARED: dict = {}  # full-sky map inherited by forked workers (copy-on-write)


def _cut(coords: np.ndarray) -> np.ndarray:
    """
    Extract thumbnails from the shared map (worker function).

    Parameters
    ----------
    coords : np.ndarray
        ``(n, 2)`` ``(dec, ra)`` in radians.

    Returns
    -------
    thumbs : np.ndarray
        ``(n, ny, nx)`` float32 thumbnails.
    """
    m, kw = _SHARED["map"], _SHARED["kw"]
    return np.asarray(reproject.thumbnails(m, coords, proj="tan", **kw), np.float32)


def tile_centres(
    ra: np.ndarray, dec: np.ndarray, offset_arcmin: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tile centres randomly offset from cluster positions by up to ``offset_arcmin`` on the sky.

    Parameters
    ----------
    ra, dec : np.ndarray
        Cluster positions in degrees.
    offset_arcmin : float
        Maximum offset along each axis, in arcmin of true angle.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    ra_c, dec_c : np.ndarray
        Tile centres in degrees.
    """
    dra, ddec = rng.uniform(-offset_arcmin, offset_arcmin, (2, len(ra))) / 60
    dec_c = np.clip(dec + ddec, -89.9, 89.9)
    return (ra + dra / np.cos(np.radians(dec_c))) % 360, dec_c


def make_tiles(
    map_paths: list[str | Path],
    ra: np.ndarray,
    dec: np.ndarray,
    out_dir: Path,
    half_width_arcmin: float = 30.0,
    res_arcmin: float = 0.5,
    chunk: int = 512,
    n_workers: int = 1,
) -> TileWCS:
    """
    Cut TAN tiles centred on ``(ra, dec)`` from each map and write them in packed format.

    Every frequency uses the same gnomonic geometry, so channels are aligned.
    Output matches :func:`mlclusters.data.pack_fits_tiles` (``images.npy`` and ``tiles.npz``).

    Parameters
    ----------
    map_paths : list[str | Path]
        One full-sky map per frequency channel.
    ra, dec : np.ndarray
        Tile centres in degrees.
    out_dir : Path
        Output directory.
    half_width_arcmin : float, default=30.0
        Tile half-width.
    res_arcmin : float, default=0.5
        Tile pixel size.
    chunk : int, default=512
        Tiles extracted per call to pixell.
    n_workers : int, default=1
        Worker processes (forked, sharing the map in memory).

    Returns
    -------
    wcs : TileWCS
        Geometry of the tiles.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    r, res = half_width_arcmin * utils.arcmin, res_arcmin * utils.arcmin
    oshape, owcs = enmap.thumbnail_geometry(r=r, res=res, proj="tan")
    oshape = tuple(int(n) for n in oshape)
    images = np.lib.format.open_memmap(
        out_dir / IMAGES, mode="w+", dtype=np.float32, shape=(len(ra), len(map_paths), *oshape)
    )
    coords = np.radians(np.stack([dec, ra], axis=1))
    starts = range(0, len(ra), chunk)
    for c, path in enumerate(map_paths):
        imap = enmap.read_map(str(path))
        _SHARED["map"] = imap.reshape(-1, *imap.shape[-2:])[0]  # temperature only
        _SHARED["kw"] = {"r": r, "res": res, "oshape": oshape, "owcs": owcs}
        with mp.get_context("fork").Pool(n_workers) as pool:
            thumbs = pool.imap(_cut, (coords[s : s + chunk] for s in starts))
            for s, th in zip(starts, thumbs, strict=True):
                images[s : s + len(th), c] = th
        print(f"channel {c} done: {path}", flush=True)
        _SHARED.clear()
        del imap
    images.flush()
    n = len(ra)
    wcs = TileWCS(
        crval1=np.asarray(ra, float),
        crval2=np.asarray(dec, float),
        crpix1=np.full(n, owcs.wcs.crpix[0]),
        crpix2=np.full(n, owcs.wcs.crpix[1]),
        cdelt1=np.full(n, owcs.wcs.cdelt[0]),
        cdelt2=np.full(n, owcs.wcs.cdelt[1]),
    )
    source = np.full(n, "make_tiles")
    np.savez(out_dir / TILES, tile_id=np.arange(n), source=source, **wcs.as_dict())
    return wcs
