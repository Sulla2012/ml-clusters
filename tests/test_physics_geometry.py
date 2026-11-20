"""Tests for halo sizes, sky geometry and catalogue matching."""

import numpy as np
import pytest
from astropy import wcs
from astropy.io import fits

from mlclusters.geometry import (
    TileWCS,
    dedupe,
    erode_hpx,
    healpix_split,
    match_catalogs,
    one_to_one_match,
)
from mlclusters.halos import convert_mass, mask_radius, radius_from_mass, theta_delta


def test_convert_mass_matches_colossus():
    colossus = pytest.importorskip("colossus")  # noqa: F841
    from colossus.cosmology import cosmology
    from colossus.halo import mass_defs

    from mlclusters.halos import concentration_duffy08

    cosmology.setCosmology(
        "websky",
        {
            "flat": True,
            "H0": 68,
            "Om0": 0.31,
            "Ob0": 0.049,
            "sigma8": 0.81,
            "ns": 0.965,
            "relspecies": False,
        },
    )
    m, z = np.array([2e14, 1e15]), np.array([0.3, 1.0])
    c = concentration_duffy08(m, z)
    m500, _ = convert_mass(m, z, "200m", "500c")
    for i in range(2):
        ref, _, _ = mass_defs.changeMassDefinition(m[i] * 0.68, c[i], z[i], "200m", "500c")
        assert m500[i] == pytest.approx(ref / 0.68, rel=1e-3)


def test_identity_conversion_and_scaling():
    m, z = np.array([3e14]), np.array([0.5])
    m_out, r_out = convert_mass(m, z, "500c", "500c")
    assert m_out[0] == m[0]
    assert r_out[0] == pytest.approx(radius_from_mass(m, z, "500c")[0])
    # Bigger mass -> bigger angular size; higher z (beyond z~1.6 d_A turnover aside) -> smaller.
    t = theta_delta(np.array([2e14, 2e15]), np.array([0.5, 0.5]))
    assert t[1] / t[0] == pytest.approx(10 ** (1 / 3), rel=0.1)
    assert (
        theta_delta(np.array([5e14]), np.array([0.2]))[0]
        > theta_delta(np.array([5e14]), np.array([0.8]))[0]
    )
    assert np.allclose(mask_radius(np.array([0.1, 3.0, 50.0])), [1.0, 3.0, 6.0])


@pytest.mark.parametrize("dec0", [-75.0, -10.0, 40.0])
def test_tilewcs_matches_astropy(dec0):
    h = fits.Header()
    h.update(
        NAXIS=2,
        NAXIS1=120,
        NAXIS2=120,
        CTYPE1="RA---TAN",
        CTYPE2="DEC--TAN",
        CRVAL1=359.8,
        CRVAL2=dec0,
        CRPIX1=61.0,
        CRPIX2=62.0,
        CDELT1=-1 / 120,
        CDELT2=1 / 120,
        CUNIT1="deg",
        CUNIT2="deg",
    )
    ref, tw = wcs.WCS(h), TileWCS.from_headers([h])
    x, y = np.random.default_rng(0).uniform(-10, 130, (2, 100))
    ra, dec = ref.wcs_pix2world(x, y, 0)
    ra2, dec2 = tw.pix2world(x, y, 0)
    assert np.allclose((ra - ra2 + 180) % 360 - 180, 0, atol=1e-10)
    assert np.allclose(dec, dec2, atol=1e-10)
    xb, yb = tw.world2pix(ra, dec, 0)
    assert np.allclose(xb, x, atol=1e-8) and np.allclose(yb, y, atol=1e-8)


def test_matching_handles_ra_wrap_and_cos_dec():
    # 0.5' apart across RA=0, and 1 deg of RA at dec=-80 (~10.4' on sky).
    idx, sep = match_catalogs(
        np.array([359.996, 10.0]),
        np.array([0.0, -80.0]),
        np.array([0.0045, 11.0]),
        np.array([0.0, -80.0]),
        12.0,
    )
    assert idx.tolist() == [0, 1]
    assert sep[0] == pytest.approx(0.51, abs=0.01)
    assert sep[1] == pytest.approx(60 * np.cos(np.radians(80)), rel=1e-3)


def test_one_to_one_match_counts_duplicates_as_false():
    ra_t, dec_t = np.array([10.0]), np.array([0.0])
    ra_p, dec_p = np.array([10.0, 10.01]), np.array([0.0, 0.0])
    match = one_to_one_match(ra_p, dec_p, np.array([0.5, 0.9]), ra_t, dec_t, 2.0)
    assert sorted(match.tolist()) == [-1, 0]
    assert match[1] == 0  # higher score claims the truth


def test_dedupe_keeps_highest():
    keep = dedupe(np.array([10.0, 10.01, 20.0]), np.zeros(3), np.array([0.2, 0.8, 0.5]), 2.0)
    assert keep.tolist() == [1, 2]


def test_healpix_split_fractions():
    split, regions = healpix_split(np.zeros(1), np.zeros(1), (0.8, 0.1, 0.1), nside=4)
    counts = np.bincount(regions, minlength=3) / len(regions)
    assert np.allclose(counts, [0.8, 0.1, 0.1], atol=0.01)


def test_erode_hpx_shrinks_disc() -> None:
    """Eroding a 1 deg disc by 10' leaves roughly a 50' disc, all inside the original."""
    import healpy as hp

    nside = 2048
    v = hp.ang2vec(30.0, -20.0, lonlat=True)
    pix = np.sort(hp.query_disc(nside, v, np.radians(1.0)))
    out = erode_hpx(pix, nside, 10.0, nside_work=1024)
    assert np.isin(out, pix).all()
    r_eff = np.degrees(np.sqrt(len(out) * hp.nside2pixarea(nside) / np.pi)) * 60
    assert 48 < r_eff < 51.5
