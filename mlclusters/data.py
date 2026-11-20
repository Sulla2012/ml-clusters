"""Packed tile storage and the PyTorch dataset used for training and inference."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from .geometry import TileWCS
from .masks import MaskConfig, TileTargets, tile_training_maps

IMAGES = "images.npy"
TILES = "tiles.npz"
TARGETS = "targets.npz"


def _read_fits_tile(path: str) -> tuple[np.ndarray, fits.Header]:
    """
    Read one FITS tile.

    Parameters
    ----------
    path : str
        Path to a ``(n_freq, ny, nx)`` FITS image.

    Returns
    -------
    data : np.ndarray
        Image as ``float32``.
    header : fits.Header
        Primary header.
    """
    with fits.open(path) as hdul:
        return hdul[0].data.astype(np.float32), hdul[0].header


def pack_fits_tiles(
    paths: list[Path], out_dir: Path, n_workers: int = 8, chunk: int = 2048
) -> TileWCS:
    """
    Pack many FITS tiles into one ``images.npy`` memmap plus a ``tiles.npz`` table.

    Parameters
    ----------
    paths : list[Path]
        FITS tiles, all with the same shape. The file stem must be an integer id.
    out_dir : Path
        Output directory.
    n_workers : int, default=8
        Number of reader processes.
    chunk : int, default=2048
        Tiles read per batch.

    Returns
    -------
    wcs : TileWCS
        Geometry of the packed tiles.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    first, _ = _read_fits_tile(str(paths[0]))
    images = np.lib.format.open_memmap(
        out_dir / IMAGES, mode="w+", dtype=np.float32, shape=(len(paths), *first.shape)
    )
    headers = []
    with ProcessPoolExecutor(n_workers) as pool:
        for start in range(0, len(paths), chunk):
            batch = [str(p) for p in paths[start : start + chunk]]
            for i, (img, hdr) in enumerate(pool.map(_read_fits_tile, batch, chunksize=64)):
                images[start + i] = img
                headers.append(hdr)
            print(f"packed {min(start + chunk, len(paths))}/{len(paths)}", flush=True)
    images.flush()
    wcs = TileWCS.from_headers(headers)
    np.savez(
        out_dir / TILES,
        tile_id=np.array([int(p.stem) for p in paths]),
        source=np.array([p.parent.name for p in paths]),
        **wcs.as_dict(),
    )
    return wcs


class PackedTiles:
    """
    Read-only view of a packed tile directory.

    Parameters
    ----------
    root : str | Path
        Directory containing ``images.npy``, ``tiles.npz`` and optionally
        ``targets.npz``.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.images = np.load(self.root / IMAGES, mmap_mode="r")
        with np.load(self.root / TILES) as d:
            self.tile_id = d["tile_id"]
            self.source = d["source"]
            self.wcs = TileWCS(*[d[k] for k in TileWCS.__dataclass_fields__])
        tpath = self.root / TARGETS
        self.targets = TileTargets.load(tpath) if tpath.exists() else None

    def __len__(self) -> int:
        """
        Number of tiles.

        Returns
        -------
        n : int
            Number of tiles.
        """
        return len(self.images)

    @property
    def shape(self) -> tuple[int, int]:
        """
        Spatial shape ``(ny, nx)`` of every tile.

        Returns
        -------
        shape : tuple[int, int]
            Tile shape.
        """
        return tuple(self.images.shape[-2:])

    @property
    def pix_arcmin(self) -> float:
        """
        Pixel size in arcmin (tiles share one resolution).

        Returns
        -------
        pix : float
            Pixel size.
        """
        return float(np.abs(self.wcs.cdelt2[0]) * 60)


def robust_stats(images: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-channel median and MAD-based standard deviation over a sample of tiles.

    Parameters
    ----------
    images : np.ndarray
        ``(n, c, ny, nx)`` array (may be a memmap).
    idx : np.ndarray
        Tiles to sample.

    Returns
    -------
    center, scale : np.ndarray
        ``(c,)`` arrays.
    """
    x = np.asarray(images[np.sort(idx)], np.float64)
    x = np.moveaxis(x, 1, 0).reshape(x.shape[1], -1)
    center = np.median(x, axis=1)
    scale = 1.4826 * np.median(np.abs(x - center[:, None]), axis=1)
    return center.astype(np.float32), scale.astype(np.float32)


class Normalizer:
    """
    Fixed per-channel normalisation with soft (asinh) compression of outliers.

    The per-tile median is removed, the map is divided by a global per-channel
    scale (so relative amplitudes between tiles and frequencies are kept), and
    ``k * asinh(x / k)`` keeps |x| <~ k linear while taming bright point sources.

    Parameters
    ----------
    scale : np.ndarray
        ``(c,)`` per-channel scales in map units.
    k : float, default=3.0
        Linear range in units of ``scale``; ``k <= 0`` disables compression.
    subtract_median : bool, default=True
        Remove the per-tile, per-channel median first.
    channels : list[int] or None, default=None
        Input channels to keep (e.g. ``[0, 1]`` for f090 + f150), selected before anything else;
        ``scale`` refers to the kept channels. ``None`` keeps all.
    """

    def __init__(
        self,
        scale: np.ndarray,
        k: float = 3.0,
        subtract_median: bool = True,
        channels: list[int] | None = None,
    ) -> None:
        self.scale = np.asarray(scale, np.float32)
        self.k = float(k)
        self.subtract_median = subtract_median
        self.channels = None if channels is None else [int(c) for c in channels]

    @classmethod
    def identity(cls, n_chan: int) -> Normalizer:
        """
        A no-op normaliser (raw map units), as used by the legacy pipeline.

        Parameters
        ----------
        n_chan : int
            Number of channels.

        Returns
        -------
        norm : Normalizer
            Identity normaliser.
        """
        return cls(np.ones(n_chan), k=0.0, subtract_median=False)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Normalise one ``(c, ny, nx)`` image.

        Parameters
        ----------
        img : np.ndarray
            Image in map units.

        Returns
        -------
        out : np.ndarray
            Normalised ``float32`` image.
        """
        if self.channels is not None:
            img = img[self.channels]
        if self.subtract_median:
            img = img - np.median(img, axis=(-2, -1), keepdims=True)
        img = img / self.scale[:, None, None]
        if self.k > 0:
            img = self.k * np.arcsinh(img / self.k)
        return img.astype(np.float32)

    def state(self) -> dict[str, list[float] | float]:
        """
        JSON-serialisable state.

        Returns
        -------
        state : dict
            Constructor arguments.
        """
        return {
            "scale": self.scale.tolist(),
            "k": self.k,
            "subtract_median": self.subtract_median,
            "channels": self.channels,
        }


def dihedral(arrs: list[np.ndarray], k: int) -> list[np.ndarray]:
    """
    Apply one of the 8 dihedral symmetries to the last two axes of each array.

    Parameters
    ----------
    arrs : list[np.ndarray]
        Arrays sharing their trailing ``(ny, nx)`` axes.
    k : int
        Symmetry index in ``[0, 8)``: ``k % 4`` quarter turns, then a flip if ``k >= 4``.

    Returns
    -------
    out : list[np.ndarray]
        Transformed (contiguous) arrays.
    """
    out = []
    for a in arrs:
        a = np.rot90(a, k % 4, axes=(-2, -1))
        if k >= 4:
            a = a[..., ::-1]
        out.append(np.ascontiguousarray(a))
    return out


class TileDataset(torch.utils.data.Dataset):
    """
    Training/inference dataset over a subset of packed tiles.

    Parameters
    ----------
    tiles : PackedTiles
        Packed tile directory (must contain targets for training).
    idx : np.ndarray
        Tile indices in this split.
    norm : Normalizer
        Input normalisation.
    cfg : MaskConfig
        Target configuration.
    augment : bool, default=False
        Apply random dihedral transforms.
    with_targets : bool, default=True
        Return dense targets (set False for pure inference).
    """

    def __init__(
        self,
        tiles: PackedTiles,
        idx: np.ndarray,
        norm: Normalizer,
        cfg: MaskConfig,
        augment: bool = False,
        with_targets: bool = True,
    ) -> None:
        self.tiles, self.idx, self.norm, self.cfg = tiles, np.asarray(idx), norm, cfg
        self.augment, self.with_targets = augment, with_targets

    def __len__(self) -> int:
        """
        Number of tiles in the split.

        Returns
        -------
        n : int
            Length.
        """
        return len(self.idx)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """
        Load, normalise and (optionally) augment one tile with its targets.

        Parameters
        ----------
        i : int
            Index within the split.

        Returns
        -------
        sample : dict[str, torch.Tensor]
            ``image`` plus, if requested, ``heat``, ``radius``, ``reg_mask``,
            ``weight``; ``index`` is the packed tile index.
        """
        t = int(self.idx[i])
        img = self.norm(np.asarray(self.tiles.images[t]))
        out = {"index": torch.tensor(t)}
        arrs, keys = [img], ["image"]
        if self.with_targets:
            maps = tile_training_maps(
                self.tiles.targets.tile(t), self.tiles.shape, self.tiles.pix_arcmin, self.cfg
            )
            keys += list(maps)
            arrs += [m[None] for m in maps.values()]
        if self.augment:
            arrs = dihedral(arrs, int(np.random.randint(8)))
        for k, a in zip(keys, arrs, strict=True):
            out[k] = torch.from_numpy(np.ascontiguousarray(a))
        return out


def save_json(path: str | Path, obj: dict) -> None:
    """
    Write a dict (dataclasses are converted) to JSON.

    Parameters
    ----------
    path : str | Path
        Output path.
    obj : dict
        Object to save.
    """
    obj = {k: asdict(v) if hasattr(v, "__dataclass_fields__") else v for k, v in obj.items()}
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
