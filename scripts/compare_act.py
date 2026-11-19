"""Put ClusterNet test results on the footing of the ACT DR5 SZ catalogue (Hilton et al. 2021).

ACT DR5 quotes a 90% completeness limit of M500c > 3.8e14 Msun (UPP-scaled masses) at z = 0.5 for
SNR2.4 > 5, where the false-detection fraction is ~3% (75 false / 13,211 deg^2 in signal-free
sims; 2471 at SNR > 4). This script measures, for each evaluation and score threshold:

- ``M90``: the true M500c at which completeness first reaches 90%, for 0.3 < z < 0.7;
- the density of detections not matched to any M200m > 1e14 halo [per deg^2] over the union
  of the (strict-footprint) test tiles, which transfers to a blind survey, unlike purity on
  cluster-centred tiles;
- purity counting any matched M200m > 1e14 halo as real.

Writes a JSON summary and a figure (completeness vs M500c at the chosen operating points).
"""

import argparse as argp
import json
from pathlib import Path

import healpy as hp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mlclusters.catalogs import Catalog  # noqa: E402
from mlclusters.data import PackedTiles  # noqa: E402

ACT_M90 = 3.8e14  # Hilton+21, M500c^UPP, z = 0.5, SNR2.4 > 5
ACT_AREA = 13211.0  # deg^2
ACT_FALSE = {"SNR>4": 2471 / ACT_AREA, "SNR>5": 75 / ACT_AREA, "SNR>6": 2 / ACT_AREA}
M_EDGES = np.geomspace(1.2e14, 2e15, 19)


def footprint_area(tiles: PackedTiles, idx: np.ndarray, edge: float, nside: int = 4096) -> float:
    """
    Area of the union of tile footprints (minus an edge margin).

    Parameters
    ----------
    tiles : PackedTiles
        Packed tiles.
    idx : np.ndarray
        Tile indices.
    edge : float
        Edge margin [pix].
    nside : int, default=4096
        HEALPix resolution used to rasterise the union.

    Returns
    -------
    area : float
        Area [deg^2].
    """
    ny, nx = tiles.shape
    wcs = tiles.wcs[idx]
    hit = np.zeros(hp.nside2npix(nside), bool)
    step = hp.nside2resol(nside, arcmin=True) / tiles.pix_arcmin / 2
    for py in np.arange(edge, ny - 1 - edge, step):
        for px in np.arange(edge, nx - 1 - edge, step):
            ra, dec = wcs.pix2world(np.full(len(idx), px), np.full(len(idx), py))
            hit[hp.ang2pix(nside, ra, dec, lonlat=True)] = True
    return float(hit.sum() * hp.nside2pixarea(nside, degrees=True))


def m90(m500c: np.ndarray, detected: np.ndarray, level: float = 0.9) -> float:
    """
    Mass at which binned completeness first reaches ``level`` (log-interpolated).

    Parameters
    ----------
    m500c : np.ndarray
        True masses of the truths.
    detected : np.ndarray
        Whether each truth is detected.
    level : float, default=0.9
        Completeness level.

    Returns
    -------
    m : float
        Mass [Msun]; nan if never reached with > 10 objects per bin.
    """
    idx = np.digitize(m500c, M_EDGES) - 1
    ok = (idx >= 0) & (idx < len(M_EDGES) - 1)
    n = np.bincount(idx[ok], minlength=len(M_EDGES) - 1)
    c = np.bincount(idx[ok], detected[ok], len(M_EDGES) - 1) / np.maximum(n, 1)
    cen = np.log(np.sqrt(M_EDGES[1:] * M_EDGES[:-1]))
    good = n > 10
    cen, c = cen[good], c[good]
    above = np.where(c >= level)[0]
    if len(above) == 0:
        return float("nan")
    j = above[0]
    if j == 0:
        return float(np.exp(cen[0]))
    f = (level - c[j - 1]) / (c[j] - c[j - 1])
    return float(np.exp(cen[j - 1] + f * (cen[j] - cen[j - 1])))


def operating_points(d: dict, m500c: np.ndarray, zsel: np.ndarray, area: float) -> list[dict]:
    """
    Metrics over a grid of score thresholds.

    Parameters
    ----------
    d : dict
        Evaluation npz contents.
    m500c : np.ndarray
        True M500c of the (inner) truths.
    zsel : np.ndarray
        Truths in the redshift range used for M90.
    area : float
        Searched area [deg^2].

    Returns
    -------
    rows : list[dict]
        One dict per threshold.
    """
    inner = d["truth_inner"]
    rows = []
    for t in np.round(np.linspace(0.05, 0.99, 95), 2):
        sel = d["det_score"] >= t
        n_false = int((sel & ~d["det_real"]).sum())
        det = d["truth_score"][inner] >= t
        rows.append(
            {
                "threshold": float(t),
                "n_det": int(sel.sum()),
                "purity_ext": float((sel & d["det_real"]).sum() / max(sel.sum(), 1)),
                "false_per_deg2": n_false / area,
                "m90_z0.3-0.7": m90(m500c[zsel], det[zsel]),
                "completeness_m500c>3.8e14_z0.3-0.7": float(det[zsel & (m500c > ACT_M90)].mean()),
            }
        )
    return rows


def _make_parser() -> argp.ArgumentParser:
    """
    Build the CLI parser.

    Returns
    -------
    parser : argp.ArgumentParser
        Parser.
    """
    p = argp.ArgumentParser(description=__doc__, formatter_class=argp.RawDescriptionHelpFormatter)
    p.add_argument("tiles", type=Path, help="Packed tiles the evals were run on (for the area).")
    p.add_argument("evals", nargs="+", help="label=eval_prefix")
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--split", default="split_act.npz")
    p.add_argument("--edge", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=Path("reports/eval/act_comparison"))
    p.add_argument("--debug", action="store_true", help="Coarse area raster (nside 1024).")
    return p


def main() -> None:
    """Compute and save the comparison."""
    args = _make_parser().parse_args()
    tiles = PackedTiles(args.tiles)
    idx = np.where(np.load(args.tiles / args.split)["split"] == 2)[0]
    area = footprint_area(tiles, idx, args.edge, 1024 if args.debug else 4096)
    print(f"test area (union of tiles): {area:.0f} deg^2", flush=True)
    cat = Catalog.load(args.catalog)
    out = {"area_deg2": area, "act_false_per_deg2": ACT_FALSE, "act_m90": ACT_M90, "models": {}}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for item in args.evals:
        label, pre = item.split("=", 1)
        d = dict(np.load(Path(pre).with_suffix(".npz")))
        inner = d["truth_inner"]
        m500c = cat.m500c[d["truth_cat_idx"][inner]]
        z = d["truth_z"][inner]
        zsel = (z > 0.3) & (z < 0.7)
        rows = operating_points(d, m500c, zsel, area)
        out["models"][label] = rows
        best = next((r for r in rows if r["purity_ext"] >= 0.97), rows[-1])
        strict = next((r for r in rows if r["false_per_deg2"] <= ACT_FALSE["SNR>5"]), rows[-1])
        for r in (best, strict):
            print(
                f"{label}: t={r['threshold']:.2f} purity_ext {r['purity_ext']:.3f} "
                f"false {r['false_per_deg2']:.3f}/deg2 M90(z 0.3-0.7) {r['m90_z0.3-0.7']:.2e} "
                f"C(M500c>3.8e14) {r['completeness_m500c>3.8e14_z0.3-0.7']:.3f}",
                flush=True,
            )
        det = d["truth_score"][inner] >= best["threshold"]
        i_m = np.digitize(m500c[zsel], M_EDGES) - 1
        ok = (i_m >= 0) & (i_m < len(M_EDGES) - 1)
        n = np.bincount(i_m[ok], minlength=len(M_EDGES) - 1)
        c = np.bincount(i_m[ok], det[zsel][ok], len(M_EDGES) - 1) / np.maximum(n, 1)
        cen = np.sqrt(M_EDGES[1:] * M_EDGES[:-1])
        ax.plot(
            cen[n > 10],
            c[n > 10],
            "o-",
            ms=3,
            label=f"{label} (t = {best['threshold']:.2f}, purity {best['purity_ext']:.2f})",
        )
    ax.axhline(0.9, color="k", lw=0.5)
    ax.axvline(ACT_M90, color="k", ls="--", lw=0.8, label="ACT DR5 90% limit (M500c^UPP)")
    ax.set(
        xscale="log",
        xlabel="true M500c [Msun]",
        ylabel="completeness, 0.3 < z < 0.7",
        ylim=(0, 1.02),
        title="at the lowest threshold with purity ≥ 0.97",
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=110)
    args.out.with_suffix(".json").write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}.json/.png")


if __name__ == "__main__":
    main()
