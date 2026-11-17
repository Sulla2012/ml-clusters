"""Copy a packed tile set, shifting some channels by whole pixels to undo a misalignment.

The legacy stamper cut the 2nd/3rd frequencies with a CAR thumbnail geometry, which
left them one pixel (0.5') higher in Dec than the first channel and the stored WCS.
The default shift (+1 row for channels 1 and 2) undoes that; the edge row is replicated.
"""

import argparse as argp
import shutil
from pathlib import Path

import numpy as np

from mlclusters.data import IMAGES


def shift_rows(img: np.ndarray, dy: int) -> np.ndarray:
    """
    Shift ``(..., ny, nx)`` images by ``dy`` rows so that ``out[y] = img[y - dy]``.

    Parameters
    ----------
    img : np.ndarray
        Images.
    dy : int
        Row shift; vacated rows replicate the nearest edge row.

    Returns
    -------
    out : np.ndarray
        Shifted images.
    """
    if dy == 0:
        return img
    out = np.roll(img, dy, axis=-2)
    if dy > 0:
        out[..., :dy, :] = img[..., :1, :]
    else:
        out[..., dy:, :] = img[..., -1:, :]
    return out


def _make_parser() -> argp.ArgumentParser:
    """
    Build the CLI parser.

    Returns
    -------
    parser : argp.ArgumentParser
        Parser.
    """
    p = argp.ArgumentParser(description=__doc__, formatter_class=argp.RawTextHelpFormatter)
    p.add_argument("src", type=Path, help="Packed tile directory to fix.")
    p.add_argument("dst", type=Path, help="Output packed tile directory.")
    p.add_argument("--channels", type=int, nargs="+", default=[1, 2], help="Channels to shift.")
    p.add_argument("--dy", type=int, default=1, help="Row shift applied to those channels.")
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--debug", action="store_true", help="Only process the first 1000 tiles.")
    return p


def main() -> None:
    """Write the realigned copy (images plus the unchanged metadata files)."""
    args = _make_parser().parse_args()
    src = np.load(args.src / IMAGES, mmap_mode="r")
    n = 1000 if args.debug else len(src)
    args.dst.mkdir(parents=True, exist_ok=True)
    dst = np.lib.format.open_memmap(args.dst / IMAGES, "w+", np.float32, (n, *src.shape[1:]))
    for s in range(0, n, args.chunk):
        block = np.array(src[s : min(s + args.chunk, n)])
        for c in args.channels:
            block[:, c] = shift_rows(block[:, c], args.dy)
        dst[s : s + len(block)] = block
    dst.flush()
    for f in args.src.glob("*.npz"):
        if args.debug and f.name != "tiles.npz":
            continue  # targets/split are per-tile and would not match a truncated copy
        shutil.copy(f, args.dst / f.name)
    print(f"wrote {n} realigned tiles to {args.dst}")


if __name__ == "__main__":
    main()
