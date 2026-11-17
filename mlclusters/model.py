"""ConvNeXt U-Net with CenterNet-style heads for anchor-free cluster detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    """
    Architecture hyper-parameters.

    Attributes
    ----------
    in_chans : int
        Number of input frequency channels.
    dims : tuple[int, ...]
        Channel widths of the encoder stages (each stage halves the resolution).
    depths : tuple[int, ...]
        ConvNeXt blocks per encoder stage.
    dec_depth : int
        ConvNeXt blocks per decoder stage.
    drop_path : float
        Maximum stochastic-depth rate.
    mass_head : bool
        Add a head regressing log10(M200m / 1e14) at cluster centres.
    """

    in_chans: int = 3
    dims: tuple[int, ...] = field(default=(32, 64, 128, 256))
    depths: tuple[int, ...] = field(default=(2, 2, 6, 2))
    dec_depth: int = 1
    drop_path: float = 0.1
    mass_head: bool = False


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel dimension of an NCHW tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalise over channels.

        Parameters
        ----------
        x : torch.Tensor
            ``(n, c, h, w)`` input.

        Returns
        -------
        y : torch.Tensor
            Normalised tensor of the same shape.
        """
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block: 7x7 depthwise conv, LayerNorm, inverted-bottleneck MLP, layer scale.

    Parameters
    ----------
    dim : int
        Number of channels.
    drop_path : float, default=0.0
        Stochastic-depth probability.
    """

    def __init__(self, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the residual block.

        Parameters
        ----------
        x : torch.Tensor
            ``(n, c, h, w)`` input.

        Returns
        -------
        y : torch.Tensor
            Output of the same shape.
        """
        y = self.dwconv(x).permute(0, 2, 3, 1)
        y = (self.gamma * self.mlp(self.norm(y))).permute(0, 3, 1, 2)
        if self.training and self.drop_path > 0:
            keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.drop_path
            y = y * keep / (1 - self.drop_path)
        return x + y


class ClusterNet(nn.Module):
    """
    ConvNeXt encoder/decoder producing full-resolution detection maps.

    The stem keeps full resolution because clusters are only a few pixels
    across. Outputs are a centre-heatmap logit and a log-radius (pixels) map.

    Parameters
    ----------
    cfg : ModelConfig
        Architecture configuration.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dims, depths = cfg.dims, cfg.depths
        rates = torch.linspace(0, cfg.drop_path, sum(depths)).tolist()
        stem_conv = nn.Conv2d(cfg.in_chans, dims[0], 3, padding=1)
        self.stem = nn.Sequential(stem_conv, LayerNorm2d(dims[0]))
        self.down, self.enc = nn.ModuleList(), nn.ModuleList()
        self.up, self.fuse, self.dec = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i, (d, n) in enumerate(zip(dims, depths, strict=True)):
            start = sum(depths[:i])
            self.enc.append(nn.Sequential(*[ConvNeXtBlock(d, rates[start + j]) for j in range(n)]))
            if i == 0:
                continue
            prev = dims[i - 1]
            self.down.append(nn.Sequential(LayerNorm2d(prev), nn.Conv2d(prev, d, 2, stride=2)))
            # Decoder modules are stored deepest-first, matching the forward pass.
            self.up.insert(0, nn.ConvTranspose2d(d, prev, 2, stride=2))
            self.fuse.insert(0, nn.Conv2d(2 * prev, prev, 1))
            self.dec.insert(0, nn.Sequential(*[ConvNeXtBlock(prev) for _ in range(cfg.dec_depth)]))
        self.heat_head = self._head(dims[0], prior=0.01)
        self.radius_head = self._head(dims[0])
        self.mass_head = self._head(dims[0]) if cfg.mass_head else None

    @staticmethod
    def _head(dim: int, prior: float | None = None) -> nn.Sequential:
        """
        Small convolutional prediction head.

        Parameters
        ----------
        dim : int
            Input channels.
        prior : float, optional
            If given, initialise the output bias so ``sigmoid(bias) = prior``.

        Returns
        -------
        head : nn.Sequential
            The head.
        """
        head = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(), nn.Conv2d(dim, 1, 1))
        if prior is not None:
            nn.init.constant_(head[-1].bias, float(torch.logit(torch.tensor(prior))))
        return head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Predict heatmap logits and log radii.

        Inputs of any size are reflect-padded to a multiple of the total stride
        and the outputs cropped back.

        Parameters
        ----------
        x : torch.Tensor
            ``(n, in_chans, h, w)`` normalised maps.

        Returns
        -------
        out : dict[str, torch.Tensor]
            ``heat`` (logits), ``radius`` (log pixels) and, with a mass head,
            ``logm`` (log10 M200m/1e14), each ``(n, 1, h, w)``.
        """
        h, w = x.shape[-2:]
        stride = 2 ** (len(self.cfg.dims) - 1)
        ph, pw = -h % stride, -w % stride
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        skips = []
        x = self.enc[0](self.stem(x))
        for down, enc in zip(self.down, self.enc[1:], strict=True):
            skips.append(x)
            x = enc(down(x))
        for up, fuse, dec in zip(self.up, self.fuse, self.dec, strict=True):
            x = dec(fuse(torch.cat([up(x), skips.pop()], dim=1)))
        x = x[..., :h, :w]
        out = {"heat": self.heat_head(x), "radius": self.radius_head(x)}
        if self.mass_head is not None:
            out["logm"] = self.mass_head(x)
        return out


def focal_loss(
    logits: torch.Tensor,
    heat: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """
    CenterNet penalty-reduced focal loss, normalised by the number of centres.

    Parameters
    ----------
    logits : torch.Tensor
        Predicted heatmap logits.
    heat : torch.Tensor
        Target heatmap with exact 1s at cluster centres.
    weight : torch.Tensor
        Per-pixel weight; 0 excludes a pixel (ignore regions).
    alpha, beta : float
        Focal and penalty-reduction exponents.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss.
    """
    logits = logits.float()
    p = torch.sigmoid(logits)
    pos = heat.eq(1).float()
    log_p, log_1mp = F.logsigmoid(logits), F.logsigmoid(-logits)
    pos_loss = -(log_p * (1 - p) ** alpha * pos).sum()
    neg_loss = -(log_1mp * p**alpha * (1 - heat) ** beta * (1 - pos) * weight).sum()
    return (pos_loss + neg_loss) / pos.sum().clamp(min=1)


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean L1 error over the pixels where ``mask`` is 1.

    Parameters
    ----------
    pred, target, mask : torch.Tensor
        Same-shaped tensors.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss (0 if the mask is empty).
    """
    err = F.l1_loss(pred.float(), target, reduction="none") * mask
    return err.sum() / mask.sum().clamp(min=1)


def detection_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    radius_weight: float = 1.0,
    mass_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    Total loss: focal heatmap loss, L1 on log radius and (optionally) on log mass.

    Parameters
    ----------
    out : dict[str, torch.Tensor]
        Model outputs.
    batch : dict[str, torch.Tensor]
        Targets ``heat``, ``radius``, ``reg_mask``, ``weight`` (and ``logm``,
        ``mass_mask`` when the model has a mass head).
    radius_weight : float, default=1.0
        Weight of the radius term.
    mass_weight : float, default=1.0
        Weight of the mass term.

    Returns
    -------
    losses : dict[str, torch.Tensor]
        ``loss`` (total), ``heat``, ``radius`` and ``mass`` components.
    """
    heat = focal_loss(out["heat"], batch["heat"], batch["weight"])
    radius = _masked_l1(out["radius"], batch["radius"], batch["reg_mask"])
    loss = heat + radius_weight * radius
    mass = torch.zeros_like(heat)
    if "logm" in out:
        mass = _masked_l1(out["logm"], batch["logm"], batch["mass_mask"])
        loss = loss + mass_weight * mass
    return {"loss": loss, "heat": heat, "radius": radius, "mass": mass}


@torch.no_grad()
def decode(
    out: dict[str, torch.Tensor], threshold: float = 0.1, max_det: int = 100
) -> list[dict[str, torch.Tensor]]:
    """
    Turn heatmaps into detections via 3x3 local-maximum suppression.

    Positions are refined to sub-pixel accuracy with the heat-weighted centroid
    of the 3x3 neighbourhood.

    Parameters
    ----------
    out : dict[str, torch.Tensor]
        Model outputs for a batch.
    threshold : float, default=0.1
        Minimum peak score.
    max_det : int, default=100
        Maximum detections per image.

    Returns
    -------
    dets : list[dict[str, torch.Tensor]]
        Per image: ``x``, ``y`` (0-based pixels), ``score``, ``radius`` (pixels)
        and, with a mass head, ``m200m`` (Msun).
    """
    p = torch.sigmoid(out["heat"].float())
    peaks = (p == F.max_pool2d(p, 3, stride=1, padding=1)) & (p >= threshold)
    # Heat-weighted 3x3 centroid offsets.
    k = torch.tensor([-1.0, 0.0, 1.0], device=p.device)
    pn = F.unfold(p, 3, padding=1).view(p.shape[0], 9, *p.shape[-2:])
    norm = pn.sum(1, keepdim=True)
    dx = (pn * k.repeat(3).view(1, 9, 1, 1)).sum(1, keepdim=True) / norm
    dy = (pn * k.repeat_interleave(3).view(1, 9, 1, 1)).sum(1, keepdim=True) / norm
    radius = out["radius"].float().exp()
    mass = 1e14 * 10 ** out["logm"].float() if "logm" in out else None
    dets = []
    for b in range(p.shape[0]):
        iy, ix = torch.nonzero(peaks[b, 0], as_tuple=True)
        score = p[b, 0, iy, ix]
        order = torch.argsort(score, descending=True)[:max_det]
        iy, ix = iy[order], ix[order]
        det = {
            "x": ix.float() + dx[b, 0, iy, ix],
            "y": iy.float() + dy[b, 0, iy, ix],
            "score": score[order],
            "radius": radius[b, 0, iy, ix],
        }
        if mass is not None:
            det["m200m"] = mass[b, 0, iy, ix]
        dets.append(det)
    return dets
