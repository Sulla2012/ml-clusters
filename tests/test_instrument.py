"""Tests for the beam and noise model."""

import numpy as np
from pixell import enmap, utils

from mlclusters.instrument import (
    NoiseShape,
    beam_fwhm_arcmin,
    depth_map,
    floor_ivar,
    noise_map,
    patch_power,
)


def test_noise_shape_interpolates_and_clamps() -> None:
    """Log-log interpolation between nodes, constant outside, never below white."""
    s = NoiseShape([100.0, 1000.0, 10000.0], [1e4, 1e2, 0.9])
    assert np.isclose(s(np.array([np.sqrt(100 * 1000)]))[0], 1e3)
    assert np.isclose(s(np.array([10.0]))[0], 1e4)
    assert s(np.array([5e4]))[0] == 1.0


def test_gaussian_beam_fwhm() -> None:
    """A Gaussian B_ell returns its own FWHM."""
    sigma = 1.4 * utils.arcmin / np.sqrt(8 * np.log(2))
    ell = np.arange(30000)
    assert abs(beam_fwhm_arcmin(np.exp(-0.5 * ell**2 * sigma**2)) - 1.4) < 0.01


def test_depth_map_units() -> None:
    """ivar = 1/sigma^2 per 0.5' pixel at the equator gives depth sigma * 0.5 uK-arcmin."""
    shape, wcs = enmap.geometry(pos=np.deg2rad([[-1, -1], [1, 1]]), res=0.5 * utils.arcmin)
    ivar = enmap.full(shape, wcs, 1 / 20.0**2)
    assert np.allclose(np.median(depth_map(ivar)), 10.0, rtol=1e-3)


def test_white_noise_level_and_footprint() -> None:
    """White-only noise has per-pixel std 1/sqrt(ivar) and is zero where ivar = 0."""
    shape, wcs = enmap.fullsky_geometry(res=30 * utils.arcmin)
    ivar = enmap.full(shape, wcs, 0.25)
    ivar[: shape[0] // 2] = 0
    n = noise_map(ivar, NoiseShape([100.0, 1000.0], [1.0, 1.0]), lmax_red=200, seed=1)
    assert np.all(n[: shape[0] // 2] == 0)
    assert abs(np.std(n[shape[0] // 2 :]) - 2.0) < 0.05


def test_patch_power_white_level() -> None:
    """Unit white noise has flat power equal to the pixel area."""
    shape, wcs = enmap.geometry(pos=np.deg2rad([[-5, -5], [5, 5]]), res=1 * utils.arcmin)
    m = enmap.enmap(np.random.default_rng(0).standard_normal(shape), wcs)
    ps = patch_power(m, enmap.ones(shape, wcs), np.array([1000.0, 3000.0, 6000.0]))
    assert np.allclose(ps / utils.arcmin**2, 1.0, rtol=0.1)


def test_floor_ivar_zeroes_shallow_pixels() -> None:
    """Pixels below the floor become unobserved and give zero (not huge) noise."""
    shape, wcs = enmap.geometry(np.array([[-1, -1], [1, 1]]) * utils.degree, res=utils.arcmin)
    ivar = enmap.full(shape, wcs, 1e-3)
    ivar[0, :5] = 1e-38
    out = floor_ivar(ivar, 0.01)
    assert (out[0, :5] == 0).all() and np.allclose(out[1:], 1e-3) and ivar[0, 0] == 1e-38
    assert np.abs(noise_map(out, NoiseShape([100.0, 1e4], [1.0, 1.0]), 500, seed=0)).max() < 1e3
