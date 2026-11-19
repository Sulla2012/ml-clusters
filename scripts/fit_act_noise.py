"""Measure the ACT DR6 temperature noise spectrum shape from daynight-minus-night differences.

The difference ``m_daynight - m_night`` cancels the sky and has per-pixel variance
``1/ivar_night - 1/ivar_daynight``. Whitened by that, its power divided by its own white level
``A`` (from a fit of ``A [1 + (ell / ell_knee)^alpha]``) is the noise shape relative to white,
tabulated per band. ``A`` is below 1 because the daynight map is not an exact ivar-weighted
superset of the night map; the white level of the simulated noise comes from ivar directly.
Writes ``act_noise_f{band}.json`` (read by ``make_observed_maps.py``) and an npz of the spectra.
"""

import argparse as argp
from pathlib import Path

import numpy as np
from pixell import enmap, utils
from scipy.optimize import curve_fit

from mlclusters.instrument import NoiseShape, act_files, patch_power

PATCHES = [(15, -35), (35, -5), (150, 2), (175, -25), (340, -40), (5, -52), (200, 5), (60, -45)]


def _read(path: Path, box: np.ndarray) -> enmap.ndmap:
    """
    Read the temperature component of a patch.

    Parameters
    ----------
    path : Path
        ACT map or ivar (``(3, ny, nx)``).
    box : np.ndarray
        Pixell box [rad].

    Returns
    -------
    m : enmap.ndmap
        Temperature patch.
    """
    return enmap.read_map(str(path), box=box)[0]


def _model(ell: np.ndarray, log_a: float, log_knee: float, alpha: float) -> np.ndarray:
    """
    Log of the relative noise power ``A [1 + (ell / ell_knee)^alpha]``.

    Parameters
    ----------
    ell : np.ndarray
        Multipoles.
    log_a, log_knee, alpha : float
        Log amplitude, log knee and slope.

    Returns
    -------
    log_ps : np.ndarray
        Log relative power.
    """
    return log_a + np.log1p((ell / np.exp(log_knee)) ** alpha)


def measure(band: str, size_deg: float, patches: list[tuple[float, float]]) -> dict:
    """
    Whitened difference and map spectra for one band, averaged over patches.

    Parameters
    ----------
    band : str
        ACT band.
    size_deg : float
        Patch side [deg].
    patches : list[tuple[float, float]]
        Patch centres (RA, Dec) [deg].

    Returns
    -------
    res : dict
        ``ell``, ``diff`` (relative to the white level per pixel), ``map`` (whitened map
        power, same units), ``depth`` (median daynight depth [uK arcmin]) and the fit.
    """
    bins = np.r_[np.linspace(100, 1000, 10), np.geomspace(1100, 20000, 30)]
    diff, sky, depths = [], [], []
    for ra, dec in patches:
        h = size_deg / 2
        box = np.deg2rad([[dec - h, ra + h], [dec + h, ra - h]])
        m_dn = _read(act_files(band, "map", "daynight", srcfree=True), box)
        m_n = _read(act_files(band, "map", "night", srcfree=True), box)
        iv_dn = _read(act_files(band, "ivar", "daynight"), box)
        iv_n = _read(act_files(band, "ivar", "night"), box)
        ok = (iv_n > 0) & (iv_dn > 1.05 * iv_n)
        if ok.mean() < 0.8:
            print(f"  f{band} patch ({ra}, {dec}): only {ok.mean():.0%} usable, skipped")
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            w_diff = np.where(ok, 1 / np.sqrt(1 / iv_n - 1 / iv_dn), 0.0)
        pix = enmap.pixsizemap(m_dn.shape, m_dn.wcs)
        p_white = np.mean(pix)  # power of unit-variance white noise [sr]
        diff.append(patch_power((m_dn - m_n) * w_diff, ok.astype(float), bins) / p_white)
        sky.append(patch_power(m_dn * np.sqrt(iv_dn), ok.astype(float), bins) / p_white)
        depths.append(np.median(np.sqrt(pix[ok] / utils.arcmin**2 / iv_dn[ok])))
        print(f"  f{band} patch ({ra}, {dec}): depth {depths[-1]:.1f} uK-arcmin", flush=True)
    ell = np.sqrt(bins[1:] * bins[:-1])
    diff, sky = np.mean(diff, 0), np.mean(sky, 0)
    fit = ell > 200
    popt, _ = curve_fit(_model, ell[fit], np.log(diff[fit]), p0=(0.0, np.log(3000), -3.5))
    return {
        "ell": ell,
        "diff": diff,
        "map": sky,
        "depth": float(np.median(depths)),
        "white_scale": float(np.exp(popt[0])),
        "ell_knee": float(np.exp(popt[1])),
        "alpha": float(popt[2]),
        "n_patches": len(depths),
    }


def _make_parser() -> argp.ArgumentParser:
    """
    Build the CLI parser.

    Returns
    -------
    parser : argp.ArgumentParser
        Parser.
    """
    p = argp.ArgumentParser(description=__doc__)
    p.add_argument("out_dir", type=Path, help="Directory for act_noise_f*.json / .npz.")
    p.add_argument("--bands", nargs="+", default=["090", "150", "220"])
    p.add_argument("--size", type=float, default=10.0, help="Patch side [deg].")
    p.add_argument("--debug", action="store_true", help="Two 4 deg patches, f150 only.")
    return p


def main() -> None:
    """Measure and save the noise shapes."""
    args = _make_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bands, size, patches = args.bands, args.size, PATCHES
    if args.debug:
        bands, size, patches = ["150"], 4.0, PATCHES[:2]
    for band in bands:
        res = measure(band, size, patches)
        print(
            f"f{band}: depth {res['depth']:.1f} uK-arcmin, white x{res['white_scale']:.2f}, "
            f"ell_knee {res['ell_knee']:.0f}, alpha {res['alpha']:.2f}",
            flush=True,
        )
        rel = res["diff"] / res["white_scale"]
        NoiseShape(res["ell"].tolist(), rel.tolist()).save(
            args.out_dir / f"act_noise_f{band}.json",
            ell_knee_fit=res["ell_knee"],
            alpha_fit=res["alpha"],
            white_scale=res["white_scale"],
            depth_uk_arcmin=res["depth"],
            n_patches=res["n_patches"],
            source="ACT DR6 daynight - night srcfree difference",
        )
        np.savez(args.out_dir / f"act_noise_f{band}.npz", **res)


if __name__ == "__main__":
    main()
