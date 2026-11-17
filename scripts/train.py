"""Train the ConvNeXt U-Net cluster detector on packed tiles."""

import argparse as argp
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mlclusters.data import Normalizer, PackedTiles, TileDataset, robust_stats
from mlclusters.detect import predict, score_catalog
from mlclusters.masks import MaskConfig
from mlclusters.model import ClusterNet, ModelConfig
from mlclusters.training import (
    cosine_schedule,
    heat_detector,
    run_epoch,
    save_checkpoint,
    seed_everything,
    worker_init,
)


def _make_parser() -> argp.ArgumentParser:
    """
    Build the CLI parser.

    Returns
    -------
    parser : argp.ArgumentParser
        Parser.
    """
    p = argp.ArgumentParser(description=__doc__)
    p.add_argument("tiles", type=Path, help="Packed tile directory (with targets/split).")
    p.add_argument("out_dir", type=Path, help="Directory for checkpoints and logs.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", "-b", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup", type=float, default=0.5, help="Warm-up length in epochs.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--radius_weight", type=float, default=1.0)
    p.add_argument("--dims", type=int, nargs="+", default=list(ModelConfig.dims))
    p.add_argument("--depths", type=int, nargs="+", default=list(ModelConfig.depths))
    p.add_argument("--m_cut", type=float, default=MaskConfig.m_cut, help="Positive M200m cut.")
    p.add_argument("--m_ignore", type=float, default=MaskConfig.m_ignore)
    p.add_argument("--mask_scale", type=float, default=MaskConfig.scale)
    p.add_argument("--r_min", type=float, default=MaskConfig.r_min)
    p.add_argument("--r_max", type=float, default=MaskConfig.r_max)
    p.add_argument("--sigma_frac", type=float, default=MaskConfig.sigma_frac)
    p.add_argument("--no_compile", action="store_true", help="Disable torch.compile.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="split.npz", help="Split file in the tile directory.")
    p.add_argument("--debug", action="store_true", help="Tiny run: 2 epochs, few batches.")
    return p


def main() -> None:
    """Train and checkpoint the model."""
    args = _make_parser().parse_args()
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tiles = PackedTiles(args.tiles)
    split = np.load(args.tiles / args.split)["split"]
    train_idx, val_idx = np.where(split == 0)[0], np.where(split == 1)[0]
    max_steps = None
    if args.debug:
        args.epochs, max_steps = 2, 20
        val_idx = val_idx[:256]
    print(f"{len(train_idx)} train / {len(val_idx)} val tiles on {device}")

    rng = np.random.default_rng(args.seed)
    _, scale = robust_stats(tiles.images, rng.choice(train_idx, min(500, len(train_idx)), False))
    norm = Normalizer(scale)
    cfg = MaskConfig(
        args.m_cut, args.m_ignore, args.mask_scale, args.r_min, args.r_max, args.sigma_frac
    )
    loader_kw = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "worker_init_fn": worker_init,
        "persistent_workers": args.workers > 0,
    }
    train_dl = torch.utils.data.DataLoader(
        TileDataset(tiles, train_idx, norm, cfg, augment=True),
        shuffle=True,
        drop_last=True,
        **loader_kw,
    )
    val_dl = torch.utils.data.DataLoader(TileDataset(tiles, val_idx, norm, cfg), **loader_kw)

    mcfg = ModelConfig(
        in_chans=tiles.images.shape[1], dims=tuple(args.dims), depths=tuple(args.depths)
    )
    model = ClusterNet(mcfg).to(device).to(memory_format=torch.channels_last)
    net = model if args.no_compile or device.type != "cuda" else torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = len(train_dl) if max_steps is None else max_steps
    sched = cosine_schedule(opt, args.epochs * steps, int(args.warmup * steps))

    config = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "model": asdict(mcfg),
        "mask": asdict(cfg),
        "norm": norm.state(),
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    log_path, best_f1 = args.out_dir / "log.jsonl", -1.0
    log_path.write_text("")
    for epoch in range(args.epochs):
        tr = run_epoch(net, train_dl, device, opt, sched, args.radius_weight, max_steps)
        va = run_epoch(net, val_dl, device, radius_weight=args.radius_weight)
        net.eval()
        scores = score_catalog(predict(heat_detector(net), val_dl, device), tiles, val_idx, cfg)
        thr = scores.best_threshold()
        best = {k: float(v[0]) for k, v in scores.curves(np.array([thr])).items()}
        lr = sched.get_last_lr()[0]
        rec = {"epoch": epoch, "lr": lr, "train": tr, "val": va, "val_cat": best}
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        save_checkpoint(args.out_dir / "last.pt", model, cfg, norm, epoch=epoch, val=best)
        if best["f1"] > best_f1:
            best_f1 = best["f1"]
            save_checkpoint(args.out_dir / "best.pt", model, cfg, norm, epoch=epoch, val=best)


if __name__ == "__main__":
    main()
