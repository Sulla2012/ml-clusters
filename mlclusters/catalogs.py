"""Cluster catalogues: loading WebSky halos and ACT/nemo catalogues into a common format."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import healpy as hp
import numpy as np
from astropy.io import fits

from .cosmology import RHO_CRIT0_H2, WEBSKY_COSMO, z_of_chi
from .halos import convert_mass, theta_delta


@dataclass
class Catalog:
    """
    A cluster catalogue with positions, redshifts, masses and angular sizes.

    Attributes
    ----------
    ra, dec : np.ndarray
        Coordinates in degrees, ``ra`` in [0, 360).
    z : np.ndarray
        Redshift.
    m500c : np.ndarray
        M500c in Msun.
    theta500c : np.ndarray
        Angular radius R500c / d_A in arcmin.
    m200m : np.ndarray
        M200m in Msun (NaN if unavailable).
    """

    ra: np.ndarray
    dec: np.ndarray
    z: np.ndarray
    m500c: np.ndarray
    theta500c: np.ndarray
    m200m: np.ndarray

    def __len__(self) -> int:
        """
        Number of clusters.

        Returns
        -------
        n : int
            Catalogue length.
        """
        return len(self.ra)

    def __getitem__(self, sel: np.ndarray | slice) -> Catalog:
        """
        Select a subset of the catalogue.

        Parameters
        ----------
        sel : np.ndarray | slice
            Boolean mask, index array or slice.

        Returns
        -------
        cat : Catalog
            The selected rows.
        """
        return Catalog(**{f.name: getattr(self, f.name)[sel] for f in fields(self)})

    def save(self, path: str | Path) -> None:
        """
        Save the catalogue as a compressed ``.npz``.

        Parameters
        ----------
        path : str | Path
            Output path.
        """
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path: str | Path) -> Catalog:
        """
        Load a catalogue written by :meth:`save`.

        Parameters
        ----------
        path : str | Path
            Path to the ``.npz`` file.

        Returns
        -------
        cat : Catalog
            The catalogue.
        """
        with np.load(path) as d:
            return cls(**{f.name: d[f.name] for f in fields(cls)})


def read_websky_halos(
    path: str | Path,
    m_min: float = 1e14,
    z_min: float = 0.0,
    chunk: int = 1_000_000,
    max_rows: int | None = None,
) -> Catalog:
    """
    Read the WebSky ``halos.pksc`` catalogue down to a mass threshold.

    The file is sorted by decreasing mass, so only its head is read; this avoids
    loading the full ~35 GB file.

    Parameters
    ----------
    path : str | Path
        Path to ``halos.pksc``.
    m_min : float, default=1e14
        Minimum M200m in Msun.
    z_min : float, default=0.0
        Minimum redshift.
    chunk : int, default=1_000_000
        Number of rows read per chunk.
    max_rows : int, optional
        Stop after this many rows (for quick tests).

    Returns
    -------
    cat : Catalog
        Halos with ``M200m > m_min`` and ``z > z_min``.
    """
    rho_m = RHO_CRIT0_H2 * WEBSKY_COSMO.Om0 * WEBSKY_COSMO.h**2  # comoving Msun/Mpc^3
    rows = []
    with open(path, "rb") as f:
        n_tot = int(np.fromfile(f, count=3, dtype=np.int32)[0])
        n_read = 0
        limit = n_tot if max_rows is None else min(n_tot, max_rows)
        while n_read < limit:
            block = np.fromfile(f, count=10 * min(chunk, limit - n_read), dtype=np.float32)
            block = block.reshape(-1, 10)
            n_read += len(block)
            m = 4 * np.pi / 3 * rho_m * block[:, 6].astype(np.float64) ** 3
            rows.append(block[m > m_min])
            if m[-1] <= m_min:
                break
    data = np.concatenate(rows)
    xyz = data[:, :3].astype(np.float64)
    m200m = 4 * np.pi / 3 * rho_m * data[:, 6].astype(np.float64) ** 3
    z = z_of_chi(np.linalg.norm(xyz, axis=1))
    ra, dec = hp.vec2ang(xyz, lonlat=True)
    keep = z > z_min
    m500c, _ = convert_mass(m200m[keep], z[keep], "200m", "500c")
    return Catalog(
        ra=ra[keep] % 360,
        dec=dec[keep],
        z=z[keep],
        m500c=m500c,
        theta500c=theta_delta(m200m[keep], z[keep], "200m", "500c"),
        m200m=m200m[keep],
    )


def read_fits_catalog(path: str | Path, mass_col: str = "M500c") -> Catalog:
    """
    Read a nemo/ACT style FITS catalogue (``RADeg``, ``decDeg``, ``redshift``).

    Parameters
    ----------
    path : str | Path
        Path to the FITS catalogue.
    mass_col : str, default="M500c"
        Column holding M500c in units of 1e14 Msun (nemo convention).

    Returns
    -------
    cat : Catalog
        The catalogue; clusters without a redshift get NaN sizes.
    """
    with fits.open(path) as hdul:
        d = hdul[1].data
        ra = np.asarray(d["RADeg"], float) % 360
        dec = np.asarray(d["decDeg"], float)
        names = d.columns.names
        z = np.asarray(d["redshift"], float) if "redshift" in names else np.full(len(ra), np.nan)
        m500c = (
            np.asarray(d[mass_col], float) * 1e14 if mass_col in names else np.full(len(ra), np.nan)
        )
    good = np.isfinite(z) & (z > 0) & np.isfinite(m500c) & (m500c > 0)
    theta = np.full(len(ra), np.nan)
    theta[good] = theta_delta(m500c[good], z[good], "500c", "500c")
    return Catalog(ra, dec, z, m500c, theta, np.full(len(ra), np.nan))


def load_catalog(path: str | Path, m_min: float = 1e14, z_min: float = 0.0) -> Catalog:
    """
    Load a catalogue of any supported type based on its file extension.

    Parameters
    ----------
    path : str | Path
        ``.npz`` (cached :class:`Catalog`), ``.pksc`` (WebSky) or ``.fits`` (nemo).
    m_min : float, default=1e14
        Minimum M200m for WebSky catalogues.
    z_min : float, default=0.0
        Minimum redshift for WebSky catalogues.

    Returns
    -------
    cat : Catalog
        The catalogue.
    """
    suffix = Path(path).suffix
    if suffix == ".npz":
        return Catalog.load(path)
    if suffix == ".pksc":
        return read_websky_halos(path, m_min=m_min, z_min=z_min)
    if suffix == ".fits":
        return read_fits_catalog(path)
    raise ValueError(f"Unsupported catalogue type {suffix!r}")
