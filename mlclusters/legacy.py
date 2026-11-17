"""Minimal loader for the previous Faster R-CNN (MobileNetV2) models, kept for baselines.

The old pipeline fed raw (un-normalised) maps to torchvision's Faster R-CNN; use
``Normalizer.identity`` when building the dataset for these models.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from .detect import Detector


def load_legacy_frcnn(path: str | Path, device: torch.device | str = "cpu") -> FasterRCNN:
    """
    Rebuild the legacy MobileNetV2 Faster R-CNN and load its weights.

    Parameters
    ----------
    path : str | Path
        Saved ``state_dict`` of the full detector.
    device : torch.device | str, default="cpu"
        Device.

    Returns
    -------
    model : FasterRCNN
        Model in eval mode.
    """
    backbone = torchvision.models.mobilenet_v2().features
    backbone.out_channels = 1280
    model = FasterRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=AnchorGenerator(((32, 64, 128, 256, 512),), ((0.5, 1.0, 2.0),)),
        box_roi_pool=torchvision.ops.MultiScaleRoIAlign(["0"], output_size=7, sampling_ratio=2),
    )
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def frcnn_detector(model: FasterRCNN) -> Detector:
    """
    Wrap a Faster R-CNN as a :data:`~mlclusters.detect.Detector` (box centres as positions).

    Parameters
    ----------
    model : FasterRCNN
        Model in eval mode.

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
            Raw images.

        Returns
        -------
        dets : list[dict[str, torch.Tensor]]
            Per-image detections.
        """
        out = []
        for pred in model(list(images)):
            b = pred["boxes"]
            out.append(
                {
                    "x": (b[:, 0] + b[:, 2]) / 2,
                    "y": (b[:, 1] + b[:, 3]) / 2,
                    "score": pred["scores"],
                    "radius": ((b[:, 2] - b[:, 0]) + (b[:, 3] - b[:, 1])) / 4,
                }
            )
        return out

    return detector
