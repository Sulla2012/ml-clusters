"""Tests for masks/targets, the network, decoding and catalogue scoring."""

import numpy as np
import pytest
import torch

from mlclusters.data import Normalizer, dihedral
from mlclusters.masks import MaskConfig, make_heatmap, make_mask, tile_training_maps
from mlclusters.model import ClusterNet, ModelConfig, decode, detection_loss, focal_loss


def test_make_mask_per_cluster_radius():
    mask = make_mask((100, 100), np.array([30.0, 70.0]), np.array([50.0, 50.0]), np.array([4, 12]))
    assert (mask == 1).sum() == pytest.approx(np.pi * 4**2, rel=0.15)
    assert (mask == 2).sum() == pytest.approx(np.pi * 12**2, rel=0.05)


def test_make_mask_overlap_prefers_closer_core():
    # A small cluster sitting inside a big one keeps its centre pixel.
    mask = make_mask((60, 60), np.array([30.0, 36.0]), np.array([30.0, 30.0]), np.array([15, 2]))
    assert mask[30, 36] == 2 and mask[30, 30] == 1


def test_heatmap_peaks():
    heat = make_heatmap((50, 50), np.array([10.2, 40.0]), np.array([20.0, 5.6]), np.array([2, 3]))
    assert heat[20, 10] == 1 and heat[6, 40] == 1 and (heat == 1).sum() == 2


def test_training_maps_ignore_logic():
    cfg = MaskConfig()
    rows = {
        "x": np.array([20.0, 60.0, -3.0]),
        "y": np.array([20.0, 60.0, 50.0]),
        "theta500c": np.array([2.0, 2.0, 2.0]),
        "m200m": np.array([5e14, 1.5e14, 5e14]),  # positive, sub-threshold, off-tile positive
    }
    maps = tile_training_maps(rows, (100, 100), 0.5, cfg)
    assert maps["reg_mask"].sum() == 1 and maps["heat"][20, 20] == 1
    assert maps["radius"][20, 20] == pytest.approx(np.log(4.0))  # 2' at 0.5'/pix
    w = maps["weight"]
    assert w[60, 60] == 0 and w[50, 0] == 0 and w[20, 20] == 1


def test_dihedral_consistent():
    img = np.random.default_rng(0).normal(size=(3, 8, 8))
    heat = np.zeros((1, 8, 8))
    heat[0, 1, 2] = 1
    for k in range(8):
        a, h = dihedral([img, heat], k)
        iy, ix = np.argwhere(h[0] == 1)[0]
        # The image pixel that moved with the heat peak must be the original (1, 2) pixel.
        assert np.allclose(a[:, iy, ix], img[:, 1, 2])


def test_normalizer_identity():
    img = np.random.default_rng(0).normal(size=(3, 5, 5)).astype(np.float32)
    assert np.allclose(Normalizer.identity(3)(img), img)


@pytest.mark.parametrize("shape", [(120, 120), (37, 53)])
def test_model_shapes(shape):
    model = ClusterNet(ModelConfig(dims=(8, 16, 32), depths=(1, 1, 1))).eval()
    out = model(torch.randn(2, 3, *shape))
    assert out["heat"].shape == out["radius"].shape == (2, 1, *shape)


def test_decode_subpixel_peak():
    heat = make_heatmap((40, 40), np.array([12.3]), np.array([25.0]), np.array([1.5]))
    logits = torch.logit(torch.from_numpy(heat).clamp(1e-4, 1 - 1e-4))[None, None]
    radius = torch.full_like(logits, np.log(3.0))
    det = decode({"heat": logits, "radius": radius}, threshold=0.5)[0]
    assert len(det["x"]) == 1
    assert det["x"][0].item() == pytest.approx(12.3, abs=0.25)
    assert det["y"][0].item() == pytest.approx(25.0, abs=0.05)
    assert det["radius"][0].item() == pytest.approx(3.0)


def test_losses_finite_without_positives():
    logits = torch.zeros(1, 1, 10, 10)
    heat = torch.zeros(1, 1, 10, 10)
    assert torch.isfinite(focal_loss(logits, heat, torch.ones_like(heat)))
    batch = {"heat": heat, "radius": heat, "reg_mask": heat, "weight": torch.ones_like(heat)}
    losses = detection_loss({"heat": logits, "radius": logits}, batch)
    assert torch.isfinite(losses["loss"]) and losses["radius"] == 0


def test_mass_targets_cover_subthreshold_halos():
    rows = {
        "x": np.array([20.0, 60.0]),
        "y": np.array([20.0, 60.0]),
        "theta500c": np.array([2.0, 2.0]),
        "m200m": np.array([5e14, 1.5e14]),
    }
    maps = tile_training_maps(rows, (100, 100), 0.5, MaskConfig())
    assert maps["mass_mask"].sum() == 2
    assert maps["logm"][20, 20] == pytest.approx(np.log10(5.0))
    assert maps["logm"][60, 60] == pytest.approx(np.log10(1.5))


def test_mass_head_outputs_and_decodes():
    model = ClusterNet(ModelConfig(dims=(8, 16), depths=(1, 1), mass_head=True)).eval()
    out = model(torch.randn(1, 3, 32, 32))
    assert out["logm"].shape == (1, 1, 32, 32)
    det = decode(out, threshold=0.0, max_det=3)[0]
    assert det["m200m"].shape == det["score"].shape


def test_normalizer_selects_channels() -> None:
    """``channels`` picks input bands before normalising and survives a state round trip."""
    from mlclusters.data import Normalizer

    img = np.stack([np.full((4, 4), v, np.float32) for v in (1.0, 2.0, 3.0)])
    img[:, 0, 0] += 10
    norm = Normalizer(np.array([1.0, 2.0]), k=0, channels=[0, 2])
    out = norm(img)
    assert out.shape == (2, 4, 4) and np.isclose(out[1, 0, 0], 5.0)
    again = Normalizer(**norm.state())
    assert again.channels == [0, 2] and np.allclose(again(img), out)
    assert Normalizer(np.ones(3)).state()["channels"] is None
