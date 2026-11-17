"""Training loop, checkpointing and detector wrappers for :class:`ClusterNet`."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import Normalizer
from .detect import Detector
from .masks import MaskConfig
from .model import ClusterNet, ModelConfig, decode, detection_loss


def save_checkpoint(
    path: str | Path,
    model: ClusterNet,
    mask_cfg: MaskConfig,
    norm: Normalizer,
    **extra: object,
) -> None:
    """
    Save model weights together with everything needed to reproduce inference.

    Parameters
    ----------
    path : str | Path
        Output ``.pt`` path.
    model : ClusterNet
        Model (an uncompiled module).
    mask_cfg : MaskConfig
        Target configuration used in training.
    norm : Normalizer
        Input normalisation.
    **extra : object
        Extra metadata (e.g. epoch, metrics).
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_cfg": asdict(model.cfg),
            "mask_cfg": asdict(mask_cfg),
            "norm": norm.state(),
            **extra,
        },
        path,
    )


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[ClusterNet, MaskConfig, Normalizer, dict]:
    """
    Load a checkpoint written by :func:`save_checkpoint`.

    Parameters
    ----------
    path : str | Path
        Checkpoint path.
    device : torch.device | str, default="cpu"
        Device to place the model on.

    Returns
    -------
    model : ClusterNet
        Model in eval mode.
    mask_cfg : MaskConfig
        Target configuration.
    norm : Normalizer
        Input normalisation.
    ckpt : dict
        The raw checkpoint (for metadata).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mcfg = ckpt["model_cfg"]
    mcfg = ModelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in mcfg.items()})
    model = ClusterNet(mcfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, MaskConfig(**ckpt["mask_cfg"]), Normalizer(**ckpt["norm"]), ckpt


def cosine_schedule(
    opt: torch.optim.Optimizer, total_steps: int, warmup_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warm-up followed by cosine decay to zero.

    Parameters
    ----------
    opt : torch.optim.Optimizer
        Optimiser.
    total_steps : int
        Total number of optimiser steps.
    warmup_steps : int
        Number of warm-up steps.

    Returns
    -------
    sched : torch.optim.lr_scheduler.LambdaLR
        Per-step scheduler.
    """

    def factor(step: int) -> float:
        """
        Learning-rate multiplier.

        Parameters
        ----------
        step : int
            Current step.

        Returns
        -------
        f : float
            Multiplier.
        """
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        frac = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """
    Move a batch to a device (images in channels-last layout).

    Parameters
    ----------
    batch : dict[str, torch.Tensor]
        Batch from a loader.
    device : torch.device
        Target device.

    Returns
    -------
    batch : dict[str, torch.Tensor]
        Batch on ``device``.
    """
    out = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    out["image"] = out["image"].contiguous(memory_format=torch.channels_last)
    return out


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    opt: torch.optim.Optimizer | None = None,
    sched: torch.optim.lr_scheduler.LRScheduler | None = None,
    radius_weight: float = 1.0,
    max_steps: int | None = None,
    log_every: int = 200,
) -> dict[str, float]:
    """
    One pass over a loader, training if an optimiser is given, else evaluating.

    Parameters
    ----------
    model : torch.nn.Module
        Model (possibly compiled).
    loader : torch.utils.data.DataLoader
        Batches with images and dense targets.
    device : torch.device
        Device.
    opt : torch.optim.Optimizer, optional
        Optimiser; if None the pass is evaluation-only.
    sched : torch.optim.lr_scheduler.LRScheduler, optional
        Per-step scheduler.
    radius_weight : float, default=1.0
        Weight of the radius loss.
    max_steps : int, optional
        Stop after this many batches.
    log_every : int, default=200
        Print progress every this many batches (training only).

    Returns
    -------
    losses : dict[str, float]
        Mean ``loss``, ``heat``, ``radius`` and ``mass`` over the pass.
    """
    train = opt is not None
    model.train(train)
    sums = {"loss": 0.0, "heat": 0.0, "radius": 0.0, "mass": 0.0}
    n, t0 = 0, time.time()
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = _to_device(batch, device)
        with torch.set_grad_enabled(train), torch.autocast(device.type, dtype=torch.bfloat16):
            losses = detection_loss(model(batch["image"]), batch, radius_weight)
        if train:
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
        for k in sums:
            sums[k] += losses[k].item()
        n += 1
        if train and step % log_every == 0:
            rate = (step + 1) * loader.batch_size / (time.time() - t0)
            print(
                f"  step {step:5d}/{len(loader)} loss {losses['loss'].item():.4f} "
                f"({rate:.0f} img/s)",
                flush=True,
            )
    return {k: v / max(n, 1) for k, v in sums.items()}


def heat_detector(model: torch.nn.Module, threshold: float = 0.05, max_det: int = 100) -> Detector:
    """
    Wrap a :class:`ClusterNet` as a :data:`Detector`.

    Parameters
    ----------
    model : torch.nn.Module
        Model (possibly compiled), set to eval mode by the caller.
    threshold : float, default=0.05
        Minimum peak score kept.
    max_det : int, default=100
        Maximum detections per tile.

    Returns
    -------
    detector : Detector
        Callable mapping images to detections.
    """

    def detector(images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """
        Detect clusters in a batch.

        Parameters
        ----------
        images : torch.Tensor
            Normalised images.

        Returns
        -------
        dets : list[dict[str, torch.Tensor]]
            Per-image detections.
        """
        with torch.autocast(images.device.type, dtype=torch.bfloat16):
            out = model(images.contiguous(memory_format=torch.channels_last))
        return decode(out, threshold, max_det)

    return detector


def seed_everything(seed: int) -> None:
    """
    Seed numpy and torch.

    Parameters
    ----------
    seed : int
        Seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def worker_init(worker_id: int) -> None:
    """
    Give each DataLoader worker a distinct numpy seed (for augmentation).

    Parameters
    ----------
    worker_id : int
        Worker index.
    """
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)
