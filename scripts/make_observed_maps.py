"""Make "observed" WebSky maps: ACT DR6 beam convolution plus ACT DR6-like noise.

For each band this writes, next to each other in ``out_dir``:

- ``websky_f{band}_beam.fits``        clean map convolved with the DR6 beam
- ``websky_f{band}_noise.fits``       one noise realisation (0 outside the ACT footprint)
- ``websky_f{band}_beam_noise.fits``  the sum (what the telescope would see)
- ``act_dr6_f{band}_ivar.fits``       DR6 temperature ivar on the WebSky pixelisation

The noise is white with the local DR6 ivar plus an atmospheric component whose shape comes from
``fit_act_noise.py`` (``act_noise_f{band}.json`` in ``--noise_dir``).
"""

import argparse as argp
from pathlib import Path

import numpy as np
from pixell import enmap

from mlclusters.instrument import (
    NoiseShape,
    act_files,
    beam_fwhm_arcmin,
    convolve_beam,
    load_beam,
    noise_map,
    read_ivar,
)

D = Path("/mnt/welch/USERS/jorlo/ml-clusters/maps")
CLEAN = {
    "090": Path("/mnt/welch/USERS/cwhitaker/maps/websky/websky_f090_map.fits"),
    "150": D / "websky_f150_map_cibfix.fits",
    "220": Path("/mnt/welch/USERS/cwhitaker/maps/websky/websky_f220_map.fits"),
}


def make_band(band: str, out_dir: Path, noise_dir: Path, seed: int, debug: bool) -> None:
    """
    Make and write the beam, noise, beam+noise and ivar maps for one band.

    Parameters
    ----------
    band : str
        ``"090"``, ``"150"`` or ``"220"``.
    out_dir : Path
        Output directory.
    noise_dir : Path
        Directory with ``act_noise_f{band}.json``.
    seed : int
        Noise seed (offset per band).
    debug : bool
        Work at 4 arcmin resolution.
    """
    clean = enmap.read_map(str(CLEAN[band]))
    lmax = 21600
    ivar = read_ivar(act_files(band, "ivar"), clean.shape, clean.wcs)
    if debug:
        clean, ivar, lmax = enmap.downgrade(clean, 8), enmap.downgrade(ivar, 8, op=np.sum), 2700
    bl = load_beam(act_files(band, "beam"), lmax)
    fwhm = beam_fwhm_arcmin(load_beam(act_files(band, "beam"), 30000))
    print(f"f{band}: beam FWHM ~{fwhm:.2f} arcmin, lmax {lmax}", flush=True)
    beamed = convolve_beam(clean, bl).astype(np.float32)
    del clean
    enmap.write_map(str(out_dir / f"websky_f{band}_beam.fits"), beamed)
    enmap.write_map(str(out_dir / f"act_dr6_f{band}_ivar.fits"), ivar.astype(np.float32))
    area = enmap.pixsizemap(ivar.shape, ivar.wcs)
    frac = area[ivar > 0].sum() / area.sum()
    print(f"f{band}: wrote beam map and ivar ({frac:.1%} of sky observed)", flush=True)
    shape = NoiseShape.load(noise_dir / f"act_noise_f{band}.json")
    noise = noise_map(ivar, shape, lmax_red=min(8000, lmax), seed=seed + int(band)).astype(
        np.float32
    )
    enmap.write_map(str(out_dir / f"websky_f{band}_noise.fits"), noise)
    beamed += noise
    enmap.write_map(str(out_dir / f"websky_f{band}_beam_noise.fits"), beamed)
    print(f"f{band}: wrote noise and beam+noise maps", flush=True)


def _make_parser() -> argp.ArgumentParser:
    """
    Build the CLI parser.

    Returns
    -------
    parser : argp.ArgumentParser
        Parser.
    """
    p = argp.ArgumentParser(description=__doc__, formatter_class=argp.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path, help="Output directory.")
    p.add_argument("--bands", nargs="+", default=["090", "150", "220"])
    p.add_argument("--noise_dir", type=Path, default=D, help="Where act_noise_f*.json live.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--debug", action="store_true", help="4 arcmin pixels, lmax 2700.")
    return p


def main() -> None:
    """Make the maps."""
    args = _make_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for band in args.bands:
        make_band(band, args.out_dir, args.noise_dir, args.seed, args.debug)


if __name__ == "__main__":
    main()
