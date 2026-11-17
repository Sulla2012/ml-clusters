"""Build the HTML project report: regenerate figures and fill reports/template.html."""

import argparse as argp
import base64
import io
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from mlclusters.catalogs import Catalog  # noqa: E402
from mlclusters.data import PackedTiles, TileDataset  # noqa: E402
from mlclusters.masks import MaskConfig, make_mask, radius_pix, tile_training_maps  # noqa: E402
from mlclusters.model import decode  # noqa: E402
from mlclusters.training import load_checkpoint  # noqa: E402

COLORS = ["#1f6feb", "#d1495b", "#2a9d8f", "#e9a03b", "#7b5ea7"]


def _embed(fig: plt.Figure, out_dir: Path, name: str) -> str:
    """
    Save a figure to disk and return an embeddable ``<img>`` tag.

    Parameters
    ----------
    fig : plt.Figure
        Figure.
    out_dir : Path
        Directory for the PNG copy.
    name : str
        File stem.

    Returns
    -------
    tag : str
        HTML image tag with base64 data.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img alt="{name}" src="data:image/png;base64,{data}">'


def fig_masks(tiles: PackedTiles, old_root: Path, cfg: MaskConfig) -> plt.Figure:
    """
    Old fixed-size masks versus new mass-scaled masks and heatmap targets.

    Parameters
    ----------
    tiles : PackedTiles
        Packed tiles with targets (must contain legacy training tiles 0, 1, 4).
    old_root : Path
        Directory of the legacy ``*_mask.npz`` files.
    cfg : MaskConfig
        Target configuration.

    Returns
    -------
    fig : plt.Figure
        Figure.
    """
    fig, axs = plt.subplots(3, 4, figsize=(13, 9.8))
    for row, i in enumerate([0, 1, 4]):
        rows, img = tiles.targets.tile(i), np.asarray(tiles.images[i])
        old = np.load(old_root / f"{tiles.tile_id[i]:06d}_mask.npz")["arr_0"]
        pos = rows["m200m"] >= cfg.m_cut
        r = radius_pix(rows["theta500c"], tiles.pix_arcmin, cfg)
        new = make_mask(tiles.shape, rows["x"][pos], rows["y"][pos], r[pos])
        maps = tile_training_maps(rows, tiles.shape, tiles.pix_arcmin, cfg)
        axs[row, 0].imshow(img[0], origin="lower", cmap="RdBu_r", vmin=-400, vmax=400)
        axs[row, 1].imshow(old > 0, origin="lower", cmap="gray")
        axs[row, 2].imshow(np.ma.masked_equal(new, 0), origin="lower", cmap="tab10")
        axs[row, 3].imshow(maps["heat"], origin="lower", cmap="magma")
        axs[row, 3].contour(maps["weight"], [0.5], colors="cyan", linewidths=0.7)
        titles = [
            "f090 tile [uK]",
            "old mask (fixed r = 1.2')",
            "new mask (r = θ500c, 1'–6')",
            "heatmap + ignore regions (cyan)",
        ]
        for a, t in zip(axs[row], titles, strict=True):
            a.scatter(rows["x"][pos], rows["y"][pos], marker="+", c="lime", s=30, lw=0.8)
            a.set(xlim=(-0.5, 119.5), ylim=(-0.5, 119.5), xticks=[], yticks=[], title=t)
    fig.tight_layout()
    return fig


def fig_sizes(cat: Catalog, cfg: MaskConfig) -> plt.Figure:
    """
    Distribution of theta500c for positives and the mask-radius mapping.

    Parameters
    ----------
    cat : Catalog
        Halo catalogue.
    cfg : MaskConfig
        Target configuration.

    Returns
    -------
    fig : plt.Figure
        Figure.
    """
    s = cat.m200m >= cfg.m_cut
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].hist(cat.theta500c[s], bins=np.geomspace(0.3, 15, 60), color=COLORS[0], alpha=0.8)
    axs[0].axvline(1.2, color="k", ls="--", label="old fixed radius (1.2')")
    for v in (cfg.r_min, cfg.r_max):
        axs[0].axvline(v, color=COLORS[1], ls=":")
    axs[0].set(xscale="log", xlabel="θ500c [arcmin]", ylabel="halos (M200m > 2e14)")
    axs[0].legend()
    rng = np.random.default_rng(0)
    sub = rng.choice(np.where(s)[0], 20000, replace=False)
    sc = axs[1].scatter(cat.m200m[sub], cat.theta500c[sub], c=cat.z[sub], s=2, cmap="viridis")
    axs[1].axhspan(0, cfg.r_min, color="0.85", zorder=0)
    axs[1].axhspan(cfg.r_max, 30, color="0.85", zorder=0)
    axs[1].set(
        xscale="log",
        yscale="log",
        xlabel="M200m [Msun]",
        ylabel="θ500c [arcmin]",
        ylim=(0.3, 20),
        title="grey: clipped to [r_min, r_max]",
    )
    fig.colorbar(sc, ax=axs[1], label="z")
    fig.tight_layout()
    return fig


def fig_alignment(raw: PackedTiles, fixed: PackedTiles) -> plt.Figure:
    """
    Stacked Dec profile of massive clusters per channel, before and after realignment.

    Parameters
    ----------
    raw : PackedTiles
        Original packed tiles.
    fixed : PackedTiles
        Realigned copy.

    Returns
    -------
    fig : plt.Figure
        Figure.
    """
    tg = raw.targets
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, tiles, title in ((axs[0], raw, "original tiles"), (axs[1], fixed, "realigned")):
        prof, n = np.zeros((3, 15)), 0
        for i in range(0, 3000):
            r = tg.tile(i)
            for x, y in zip(r["x"][r["m200m"] > 8e14], r["y"][r["m200m"] > 8e14], strict=True):
                ix, iy = int(round(x)), int(round(y))
                if 10 <= ix < 110 and 10 <= iy < 110:
                    st = np.asarray(tiles.images[i, :, iy - 7 : iy + 8, ix])
                    prof += st / np.abs(st[:, 7:8]).clip(1e-6)
                    n += 1
        for c, name in enumerate(["f090", "f150", "f220"]):
            ax.plot(np.arange(-7, 8), prof[c] / n, "o-", color=COLORS[c], label=name, ms=3)
        ax.axvline(0, color="k", lw=0.5)
        ax.set(xlabel="Δy from catalogue position [pix = 0.5']", title=f"{title} ({n} clusters)")
    axs[0].set_ylabel("stacked signal / |centre value|")
    axs[0].legend()
    fig.tight_layout()
    return fig


def fig_training(runs: dict[str, Path]) -> plt.Figure:
    """
    Loss and validation catalogue metrics per epoch.

    Parameters
    ----------
    runs : dict[str, Path]
        Run label to run directory (with ``log.jsonl``).

    Returns
    -------
    fig : plt.Figure
        Figure.
    """
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for k, (label, d) in enumerate(runs.items()):
        log = [json.loads(line) for line in (d / "log.jsonl").read_text().splitlines()]
        ep = [r["epoch"] for r in log]
        axs[0].plot(ep, [r["train"]["loss"] for r in log], "--", color=COLORS[k])
        axs[0].plot(ep, [r["val"]["loss"] for r in log], "-", color=COLORS[k], label=label)
        axs[1].plot(ep, [r["val_cat"]["f1"] for r in log], "-", color=COLORS[k], label=label)
        axs[1].plot(ep, [r["val_cat"]["purity"] for r in log], ":", color=COLORS[k])
        axs[1].plot(ep, [r["val_cat"]["completeness"] for r in log], "-.", color=COLORS[k])
    axs[0].set(xlabel="epoch", ylabel="loss (dashed: train, solid: val)", yscale="log")
    axs[1].set(xlabel="epoch", ylabel="val catalogue: F1 (solid), P (dotted), C (dash-dot)")
    axs[0].legend()
    fig.tight_layout()
    return fig


def fig_pc(evals: dict[str, Path]) -> plt.Figure:
    """
    Test purity and completeness versus score threshold, and the P–C curve.

    Parameters
    ----------
    evals : dict[str, Path]
        Model label to evaluation prefix (``.json``/``.npz``).

    Returns
    -------
    fig : plt.Figure
        Figure.
    """
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, (label, pre) in enumerate(evals.items()):
        res = json.loads(pre.with_suffix(".json").read_text())
        t = res["test"]
        thr = res["chosen"]["threshold_from_val"]
        axs[0].plot(t["threshold"], t["purity"], "-", color=COLORS[k], label=f"{label} purity")
        axs[0].plot(t["threshold"], t["completeness"], "--", color=COLORS[k])
        axs[0].axvline(thr, color=COLORS[k], lw=0.6)
        axs[1].plot(t["completeness"], t["purity"], "-", color=COLORS[k], label=label)
        ch = res["chosen"]["test"]
        axs[1].plot(ch["completeness"], ch["purity"], "o", color=COLORS[k])
    axs[0].set(
        xlabel="score threshold", ylabel="purity (solid) / completeness (dashed)", ylim=(0, 1.02)
    )
    axs[1].set(
        xlabel="completeness",
        ylabel="purity",
        xlim=(0, 1),
        ylim=(0, 1.02),
        title="dots: threshold chosen on validation",
    )
    axs[0].legend(fontsize=8)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_completeness(evals: dict[str, Path]) -> plt.Figure:
    """
    Test completeness versus mass and redshift at each model's chosen threshold.

    Parameters
    ----------
    evals : dict[str, Path]
        Model label to evaluation prefix.

    Returns
    -------
    fig : plt.Figure
        Figure.
    """
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    mbins, zbins = np.geomspace(2e14, 3e15, 10), np.linspace(0.05, 2.0, 11)
    for k, (label, pre) in enumerate(evals.items()):
        d = np.load(pre.with_suffix(".npz"))
        inner = d["truth_inner"]
        det = d["truth_score"][inner] >= d["threshold"]
        for ax, v, bins in (
            (axs[0], d["truth_m200m"][inner], mbins),
            (axs[1], d["truth_z"][inner], zbins),
        ):
            idx = np.digitize(v, bins) - 1
            ok = (idx >= 0) & (idx < len(bins) - 1)
            num = np.bincount(idx[ok], det[ok], len(bins) - 1)
            den = np.bincount(idx[ok], minlength=len(bins) - 1)
            cen = np.sqrt(bins[1:] * bins[:-1]) if ax is axs[0] else (bins[1:] + bins[:-1]) / 2
            good = den > 10
            ax.plot(cen[good], (num / np.maximum(den, 1))[good], "o-", color=COLORS[k], label=label)
    axs[0].set(xscale="log", xlabel="M200m [Msun]", ylabel="completeness", ylim=(0, 1.02))
    axs[1].set(xlabel="z", ylabel="completeness", ylim=(0, 1.02))
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_radius(pre: Path, pix_arcmin: float, cfg: MaskConfig) -> plt.Figure:
    """
    Predicted versus true (clipped theta500c) radius for matched detections.

    Parameters
    ----------
    pre : Path
        Evaluation prefix of a ClusterNet model.
    pix_arcmin : float
        Pixel size.
    cfg : MaskConfig
        Target configuration (for the clipping range).

    Returns
    -------
    fig : plt.Figure
        Figure.
    """
    d = np.load(pre.with_suffix(".npz"))
    m = (d["det_match"] >= 0) & (d["det_score"] >= d["threshold"])
    true = radius_pix(d["truth_theta500c"][d["det_match"][m]], pix_arcmin, cfg) * pix_arcmin
    pred = d["det_radius_pix"][m] * pix_arcmin
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.hexbin(true, pred, gridsize=40, bins="log", cmap="Blues", extent=(0.9, 6.2, 0.9, 6.2))
    ax.plot([1, 6], [1, 6], "k--", lw=0.8)
    scatter = np.std(np.log(pred / true))
    ax.set(
        xlabel="true mask radius [arcmin]",
        ylabel="predicted radius [arcmin]",
        title=f"σ(ln ratio) = {scatter:.2f}",
    )
    fig.tight_layout()
    return fig


def fig_examples(tiles: PackedTiles, ckpt: Path, split: np.ndarray) -> plt.Figure:
    """
    Example test tiles with truth (green) and detections (orange) drawn at their radii.

    Parameters
    ----------
    tiles : PackedTiles
        Packed tiles.
    ckpt : Path
        ClusterNet checkpoint.
    split : np.ndarray
        Split labels; test tiles (2) are sampled.

    Returns
    -------
    fig : plt.Figure
        Figure.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, norm, ckpt_d = load_checkpoint(ckpt, device)
    thr = ckpt_d.get("val", {}).get("threshold", 0.5)
    idx = np.random.default_rng(3).choice(np.where(split == 2)[0], 4, replace=False)
    ds = TileDataset(tiles, idx, norm, cfg, with_targets=False)
    imgs = torch.stack([ds[i]["image"] for i in range(len(idx))]).to(device)
    with torch.no_grad():
        out = model(imgs)
    dets = decode(out, thr)
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    for j, i in enumerate(idx):
        rows = tiles.targets.tile(i)
        pos = rows["m200m"] >= cfg.m_cut
        r = radius_pix(rows["theta500c"], tiles.pix_arcmin, cfg)
        axs[0, j].imshow(
            np.asarray(tiles.images[i, 0]), origin="lower", cmap="RdBu_r", vmin=-400, vmax=400
        )
        axs[1, j].imshow(
            torch.sigmoid(out["heat"][j, 0]).float().cpu(),
            origin="lower",
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        for a in axs[:, j]:
            for x, y, rr in zip(rows["x"][pos], rows["y"][pos], r[pos], strict=True):
                a.add_patch(plt.Circle((x, y), rr, fill=False, color="lime", lw=1))
            d = dets[j]
            for x, y, rr in zip(d["x"].cpu(), d["y"].cpu(), d["radius"].cpu(), strict=True):
                a.add_patch(plt.Circle((x, y), rr, fill=False, color="orange", lw=1, ls="--"))
            a.set(xlim=(-0.5, 119.5), ylim=(-0.5, 119.5), xticks=[], yticks=[])
        axs[0, j].set_title(f"test tile {tiles.tile_id[i]} (f090)")
        axs[1, j].set_title("predicted heatmap")
    fig.suptitle("green: true M200m>2e14 clusters at mask radius; orange dashed: detections")
    fig.tight_layout()
    return fig


def metrics_table(evals: dict[str, Path]) -> str:
    """
    HTML table of test metrics at the validation-chosen threshold.

    Parameters
    ----------
    evals : dict[str, Path]
        Model label to evaluation prefix.

    Returns
    -------
    html : str
        Table markup.
    """
    head = (
        "<tr><th>model</th><th>threshold</th><th>detections</th><th>purity</th>"
        "<th>purity (incl. 1e14&lt;M&lt;2e14)</th><th>completeness</th><th>F1</th></tr>"
    )
    rows = []
    for label, pre in evals.items():
        c = json.loads(pre.with_suffix(".json").read_text())["chosen"]["test"]
        rows.append(
            f"<tr><td>{label}</td><td>{c['threshold']:.2f}</td><td>{int(c['n_det'])}</td>"
            f"<td>{c['purity']:.3f}</td><td>{c['purity_ext']:.3f}</td>"
            f"<td>{c['completeness']:.3f}</td><td>{c['f1']:.3f}</td></tr>"
        )
    return f"<table>{head}{''.join(rows)}</table>"


def _pairs(items: list[str]) -> dict[str, Path]:
    """
    Parse ``label=path`` CLI items.

    Parameters
    ----------
    items : list[str]
        Items.

    Returns
    -------
    pairs : dict[str, Path]
        Label to path.
    """
    return {k: Path(v) for k, v in (s.split("=", 1) for s in items)}


def _make_parser() -> argp.ArgumentParser:
    """
    Build the CLI parser.

    Returns
    -------
    parser : argp.ArgumentParser
        Parser.
    """
    p = argp.ArgumentParser(description=__doc__)
    p.add_argument("--tiles", type=Path, required=True, help="Original packed tiles.")
    p.add_argument("--aligned", type=Path, help="Realigned packed tiles.")
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--old_masks", type=Path, required=True, help="Legacy train mask dir.")
    p.add_argument("--runs", nargs="*", default=[], help="label=run_dir")
    p.add_argument("--evals", nargs="*", default=[], help="label=eval_prefix (spatial split)")
    p.add_argument("--legacy_evals", nargs="*", default=[], help="label=prefix (legacy split)")
    p.add_argument("--example_ckpt", type=Path, help="Checkpoint for example tiles.")
    p.add_argument("--radius_eval", type=Path, help="ClusterNet eval prefix for radius plot.")
    p.add_argument("--template", type=Path, default=Path("reports/template.html"))
    p.add_argument("--out", type=Path, default=Path("reports/report.html"))
    p.add_argument("--debug", action="store_true", help="Only make the cheap figures.")
    return p


def main() -> None:
    """Make figures and write the report."""
    args = _make_parser().parse_args()
    fig_dir = args.out.parent / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tiles, cfg = PackedTiles(args.tiles), MaskConfig()
    parts = {
        "fig:masks": lambda: _embed(fig_masks(tiles, args.old_masks, cfg), fig_dir, "masks"),
        "fig:sizes": lambda: _embed(fig_sizes(Catalog.load(args.catalog), cfg), fig_dir, "sizes"),
    }
    if not args.debug:
        evals, legacy = _pairs(args.evals), _pairs(args.legacy_evals)
        split = np.load(args.tiles / "split.npz")["split"]
        parts.update(
            {
                "fig:alignment": lambda: _embed(
                    fig_alignment(tiles, PackedTiles(args.aligned)), fig_dir, "alignment"
                ),
                "fig:training": lambda: _embed(
                    fig_training(_pairs(args.runs)), fig_dir, "training"
                ),
                "fig:pc": lambda: _embed(fig_pc(evals), fig_dir, "pc"),
                "fig:pc_legacy": lambda: _embed(fig_pc(legacy), fig_dir, "pc_legacy"),
                "fig:completeness": lambda: _embed(
                    fig_completeness(evals), fig_dir, "completeness"
                ),
                "fig:completeness_legacy": lambda: _embed(
                    fig_completeness(legacy), fig_dir, "completeness_legacy"
                ),
                "fig:radius": lambda: _embed(
                    fig_radius(args.radius_eval, tiles.pix_arcmin, cfg), fig_dir, "radius"
                ),
                "fig:examples": lambda: _embed(
                    fig_examples(tiles, args.example_ckpt, split), fig_dir, "examples"
                ),
                "table:metrics": lambda: metrics_table(evals),
                "table:metrics_legacy": lambda: metrics_table(legacy),
            }
        )
    html = args.template.read_text()
    for key in re.findall(r"\{\{([a-z_:]+)\}\}", html):
        html = html.replace("{{" + key + "}}", parts[key]() if key in parts else "")
    args.out.write_text(html)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
