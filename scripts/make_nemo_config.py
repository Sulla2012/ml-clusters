"""Set up a nemo (matched-filter) cluster search over the same test sky ClusterNet was scored on.

Writes into ``out_dir``:

- ``surveyMask_<name>.fits``: uint8 CAR mask on the observed-map geometry, 1 inside the union of
  the ACT-footprint test tiles (``split_act.npz`` == 2), minus pixels shallower than
  ``--min_ivar_frac`` of the median f090 / f150 depth, minus point-source holes (``dr5``);
- ``footprint_hpx4096_<name>.npy``: the same area as sorted HEALPix pixels, for scoring ClusterNet
  on exactly the sky nemo searched (``eval_sky.py --footprint``);
- ``<name>.yml``: the nemo config. tSZ frequencies are those used to build the WebSky maps
  (93 / 145 / 225 GHz). Modes:

  - ``dr6``: nemo's ACT DR6 cluster search (98/150/220 GHz, bright + faint point-source passes,
    two cluster passes). Its bright-source pass ignores the survey mask, and on these maps it
    returns sources with absurd fluxes (1e3-1e12 Jy) in tiles whose 1 deg overlap reaches very
    shallow sky; subtracting those breaks the cluster filters of that tile and its neighbours.
  - ``ptsrc``: f150 point-source search (one beam matched filter, S/N > 5), as for the DR5 mask.
  - ``dr5``: the ACT DR5 cluster search (Hilton et al. 2021): 98 + 150 GHz, one pass, S/N > 4,
    every ``ptsrc`` source above ``--flux_cut`` masked (radius 320" rising with flux to 12') and
    filled with a smoothed map before filtering.

Run with: ``cd out_dir && mpiexec -n N ~/venvs/nemo/bin/nemo <name>.yml -M``.
"""

import argparse as argp
from pathlib import Path

import healpy as hp
import numpy as np
import yaml
from astropy.table import Table
from pixell import enmap

from mlclusters.data import PackedTiles
from mlclusters.geometry import footprint_hpx
from mlclusters.instrument import act_files

OBS = Path("/mnt/welch/USERS/jorlo/ml-clusters/maps/observed_v2")
FREQ = {"090": 93.0, "150": 145.0, "220": 225.0}
ARNAUD = [(m, z) for z in (0.2, 0.4, 0.8, 1.2) for m in (1, 2, 4, 8)]


def car_mask(hit: np.ndarray, shape: tuple[int, ...], wcs: object, chunk: int = 256) -> np.ndarray:
    """
    Sample a boolean HEALPix map at the centres of a CAR geometry.

    Parameters
    ----------
    hit : np.ndarray
        Boolean HEALPix map (RING).
    shape : tuple[int, ...]
        CAR shape.
    wcs : object
        CAR WCS.
    chunk : int, default=256
        Rows per step.

    Returns
    -------
    mask : np.ndarray
        ``uint8`` mask.
    """
    nside = hp.npix2nside(len(hit))
    ny, nx = shape[-2:]
    dec = np.rad2deg(enmap.pix2sky(shape, wcs, [np.arange(ny), np.zeros(ny)])[0])
    ra = np.rad2deg(enmap.pix2sky(shape, wcs, [np.zeros(nx), np.arange(nx)])[1])
    hp_dec = np.rad2deg(np.pi / 2 - hp.pix2ang(nside, np.where(hit)[0])[0])
    rows = np.where((dec > hp_dec.min() - 0.1) & (dec < hp_dec.max() + 0.1))[0]
    mask = np.zeros((ny, nx), np.uint8)
    for i in range(0, len(rows), chunk):
        r = rows[i : i + chunk]
        pix = hp.ang2pix(nside, ra[None, :], dec[r, None], lonlat=True)
        mask[r] = hit[pix]
    return mask


def depth_cut(hit: np.ndarray, frac: float, obs: Path) -> np.ndarray:
    """
    Drop footprint pixels shallower than ``frac`` of the median depth at f090 or f150.

    Parameters
    ----------
    hit : np.ndarray
        Boolean HEALPix footprint (RING).
    frac : float
        Minimum ivar as a fraction of the median non-zero ivar of each map.
    obs : Path
        Directory with ``act_dr6_f{band}_ivar.fits``.

    Returns
    -------
    hit : np.ndarray
        Copy with shallow pixels removed.
    """
    nside = hp.npix2nside(len(hit))
    pix = np.where(hit)[0]
    ra, dec = hp.pix2ang(nside, pix, lonlat=True)
    keep = np.ones(len(pix), bool)
    for b in ("090", "150"):
        iv = enmap.read_map(str(obs / f"act_dr6_f{b}_ivar.fits"))
        keep &= iv.at(np.deg2rad([dec, ra]), order=0) >= frac * np.median(iv[iv > 0])
    out = hit.copy()
    out[pix[~keep]] = False
    print(f"depth cut removes {np.mean(~keep):.2%} of the footprint", flush=True)
    return out


def source_holes(cat: Path, flux_cut: float, out: Path) -> Table:
    """
    DR5-style point-source mask catalogue: every source above ``flux_cut``, with a hole radius.

    The radius is 320" at the flux cut, growing as flux^(1/4) to at most 12' (DR5 masked all
    > 10 mJy sources with 320" circles after an initial 3-12' amplitude-dependent pass).

    Parameters
    ----------
    cat : Path
        nemo point-source catalogue (``fluxJy`` column), e.g. from ``--mode ptsrc``.
    flux_cut : float
        Minimum 150 GHz flux density [Jy].
    out : Path
        Where to write the masking catalogue (``RADeg``, ``decDeg``, ``fluxJy``, ``rArcmin``).

    Returns
    -------
    tab : Table
        The masking catalogue.
    """
    t = Table.read(cat)
    t = t[np.asarray(t["fluxJy"]) > flux_cut]["RADeg", "decDeg", "fluxJy"]
    t["rArcmin"] = np.clip(320 / 60 * (np.asarray(t["fluxJy"]) / flux_cut) ** 0.25, 320 / 60, 12)
    t.write(out, overwrite=True)
    print(f"{len(t)} sources > {flux_cut * 1e3:.0f} mJy, max {t['fluxJy'].max():.2f} Jy")
    return t


def punch_holes(hit: np.ndarray, holes: Table) -> np.ndarray:
    """
    Remove source holes from a HEALPix footprint.

    Parameters
    ----------
    hit : np.ndarray
        Boolean HEALPix footprint (RING).
    holes : Table
        ``RADeg``, ``decDeg``, ``rArcmin``.

    Returns
    -------
    hit : np.ndarray
        Copy with holes removed.
    """
    nside = hp.npix2nside(len(hit))
    out = hit.copy()
    vec = hp.ang2vec(np.asarray(holes["RADeg"]), np.asarray(holes["decDeg"]), lonlat=True)
    for v, r in zip(vec, np.radians(np.asarray(holes["rArcmin"]) / 60), strict=True):
        out[hp.query_disc(nside, v, r, inclusive=False)] = False
    print(f"source holes remove {1 - out.sum() / hit.sum():.2%} of the footprint", flush=True)
    return out


def tile_boxes(hit: np.ndarray, width: float = 10.0, height: float = 5.0) -> list[dict]:
    """
    Explicit nemo tiles: an RA/Dec grid of boxes, keeping those that touch the footprint.

    Nemo's autotiler drops mask regions less than ~8 deg tall, so tiles are given explicitly.
    RA edges are multiples of ``width`` from 0, so no box crosses RA = 0.

    Parameters
    ----------
    hit : np.ndarray
        Boolean HEALPix footprint (RING).
    width, height : float
        Box size [deg].

    Returns
    -------
    tiles : list[dict]
        ``tileName`` / ``RADecSection`` dicts.
    """
    nside = hp.npix2nside(len(hit))
    ra, dec = hp.pix2ang(nside, np.where(hit)[0], lonlat=True)
    i_ra = np.floor(ra / width).astype(int)
    i_dec = np.floor((dec + 90) / height).astype(int)
    out = []
    for a, b in sorted(set(zip(i_ra.tolist(), i_dec.tolist(), strict=True))):
        ra0, dec0 = a * width, b * height - 90
        out.append(
            {
                "tileName": f"t{a}_{b}",
                "RADecSection": [ra0, min(ra0 + width, 359.999), dec0, dec0 + height],
            }
        )
    return out


def config(
    mode: str, mask: str, tiles: list[dict], debug: bool, holes: str | None, obs: Path
) -> dict:
    """
    Nemo configuration, adapted to the simulated maps.

    Parameters
    ----------
    mode : str
        ``dr6``, ``dr5`` or ``ptsrc`` (see the module docstring).
    mask : str
        Survey mask file name (relative to the run directory).
    tiles : list[dict]
        Explicit tile definitions.
    debug : bool
        Two Arnaud filters only.
    holes : str or None
        Point-source masking catalogue (``dr5``).
    obs : Path
        Directory with the observed maps and ivars.

    Returns
    -------
    cfg : dict
        Config for ``yaml.safe_dump``.
    """
    bands = {"dr6": ("090", "150", "220"), "dr5": ("090", "150"), "ptsrc": ("150",)}[mode]
    maps = [
        {
            "mapFileName": str(obs / f"websky_f{b}_beam_noise.fits"),
            "weightsFileName": str(obs / f"act_dr6_f{b}_ivar.fits"),
            "obsFreqGHz": FREQ[b],
            "units": "uK",
            "beamFileName": str(act_files(b, "beam")),
            "label": f"f{b}",
        }
        for b in bands
    ]
    common = {
        "unfilteredMaps": maps,
        "surveyMask": mask,
        "minObjPix": 1,
        "findCenterOfMass": True,
        "useInterpolator": True,
        "rejectBorder": 0,
        "longNames": False,
        "useTiling": True,
        "stitchTiles": False,
        "tileOverlapDeg": 1.0,
        "tileDefinitions": tiles,
    }
    quiet = {"saveFilteredMaps": False, "saveRMSMap": False, "savePlots": False}
    if mode == "ptsrc":
        beam = {"noiseParams": {"method": "dataMap", "noiseGridArcmin": 40.0, "numNoiseBins": 8}}
        return common | {
            "thresholdSigma": 5.0,
            "objIdent": "WS-S",
            "ringThresholdSigma": 3,
            "mapFilters": [
                {
                    "label": "Beam150",
                    "class": "BeamMatchedFilter",
                    "params": beam | quiet | {"outputUnits": "uK", "edgeTrimArcmin": 0.0},
                }
            ],
        }
    arnaud = [(2, 0.4), (8, 0.4)] if debug else ARNAUD
    arnaud_filters = [
        {
            "label": f"Arnaud_M{m}e14_z{str(z).replace('.', 'p')}",
            "params": {"M500MSun": float(m) * 1e14, "z": z},
        }
        for m, z in arnaud
    ]
    if mode == "dr5":
        return common | {
            "maskPointSourcesFromCatalog": [holes],
            "thresholdSigma": 4.0,
            "objIdent": "WS-CL",
            "optimalCatalogMethod": "ref-forced",
            "catalogCuts": ["fixed_SNR > 4"],
            "photFilter": "Arnaud_M2e14_z0p4",
            "allFilters": {
                "class": "ArnaudModelMatchedFilter",
                "params": quiet
                | {
                    "noiseParams": {"method": "dataMap", "noiseGridArcmin": 40.0},
                    "saveDS9Regions": False,
                    "outputUnits": "yc",
                    "edgeTrimArcmin": 0.0,
                },
            },
            "mapFilters": arnaud_filters,
        }
    sets = {}
    for k, b in enumerate(("090", "150", "220")):
        bright, faint = 2 * k + 1, 2 * k + 2
        sets[bright] = {
            "mapToUse": f"f{b}",
            "thresholdSigma": 50.0,
            "objIdent": "SRC",
            "measureShapes": True,
            "ignoreSurveyMask": True,
        }
        sets[faint] = {
            "mapToUse": f"f{b}",
            "thresholdSigma": 5.0,
            "objIdent": "SRC",
            "subtractModelFromSets": [bright],
            "addSiphonedFromSets": [bright],
            "maskAndFillFromSets": [bright],
            "maskHoleDilationFactor": 3.0,
            "saveCatalog": True,
            "saveFilteredMaps": False,
            "fileLabel": f"ptsrc_f{b}",
        }
    sets[10] = {
        "subtractModelFromSets": [2, 4, 6],
        "maskAndFillFromSets": [1, 3, 5],
        "maskHoleDilationFactor": 3.0,
    }
    sets[11] = {**sets[10], "noiseModelCatalogFromSets": [10]}
    filters = [
        {
            "label": "BeamPass1",
            "class": "BeamMatchedFilter",
            "params": {"outputUnits": "uK", "noiseParams": {"method": "model"}},
            "filterSets": [1, 3, 5],
        },
        {
            "label": "BeamPass2",
            "class": "BeamMatchedFilter",
            "params": {"outputUnits": "uK", "noiseParams": {"method": "dataMap"}},
            "filterSets": [2, 4, 6],
        },
    ] + arnaud_filters
    return common | {
        "thresholdSigma": 3.0,
        "optimalCatalogMethod": "ref-first",
        "objIdent": "WS-CL",
        "catalogCuts": ["fixed_SNR > 3.5"],
        "flagRings": True,
        "ringFlagDistArcmin": 12.0,
        "ringFlagMinRingerSNR": 20,
        "ringThresholdSigma": 0.5,
        "photFilter": "Arnaud_M2e14_z0p4",
        "filterSetOptions": sets,
        "allFilters": {
            "class": "ArnaudModelMatchedFilter",
            "params": {
                "noiseParams": {"method": "dataMap", "noiseGridArcmin": 80.0},
                "saveFilteredMaps": False,
                "saveRMSMap": False,
                "savePlots": False,
                "saveDS9Regions": False,
                "outputUnits": "yc",
                "edgeTrimArcmin": 0.0,
            },
            "filterSets": [10, 11],
        },
        "mapFilters": filters,
    }


def _make_parser() -> argp.ArgumentParser:
    """
    Build the CLI parser.

    Returns
    -------
    parser : argp.ArgumentParser
        Parser.
    """
    p = argp.ArgumentParser(description=__doc__, formatter_class=argp.RawDescriptionHelpFormatter)
    p.add_argument("tiles", type=Path, help="Packed tiles with split_act.npz (test footprint).")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--name", default="WebSkyACTTest")
    p.add_argument("--edge", type=float, default=2.0, help="Tile edge margin [pix].")
    p.add_argument("--obs", type=Path, default=OBS, help="Observed-map directory.")
    p.add_argument("--mode", choices=("dr6", "dr5", "ptsrc"), default="dr5")
    p.add_argument("--ptsrc_cat", type=Path, help="Point-source catalogue for --mode dr5.")
    p.add_argument("--flux_cut", type=float, default=0.01, help="Source mask flux cut [Jy].")
    p.add_argument("--min_ivar_frac", type=float, default=0.1, help="Depth cut (0 = none).")
    p.add_argument("--debug", action="store_true", help="~6 deg patch, two Arnaud filters.")
    return p


def main() -> None:
    """Write the mask and config."""
    args = _make_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name + ("Debug" if args.debug else "")
    tiles = PackedTiles(args.tiles)
    idx = np.where(np.load(args.tiles / "split_act.npz")["split"] == 2)[0]
    if args.debug:
        ra, dec = tiles.wcs.crval1[idx], tiles.wcs.crval2[idx]
        d = np.hypot((ra - ra[0] + 180) % 360 - 180, dec - dec[0])
        idx = idx[d < 3.0]
    hit = footprint_hpx(tiles.wcs[idx], tiles.shape, args.edge, 4096, tiles.pix_arcmin)
    print(f"{len(idx)} tiles, {hit.sum() * hp.nside2pixarea(4096, degrees=True):.0f} deg^2")
    boxes = tile_boxes(hit)
    if args.min_ivar_frac > 0:
        hit = depth_cut(hit, args.min_ivar_frac, args.obs)
    holes = None
    if args.mode == "dr5":
        if args.ptsrc_cat is None:
            raise SystemExit("--mode dr5 needs --ptsrc_cat")
        holes = f"ptsrcMask_{name}.fits"
        hit = punch_holes(hit, source_holes(args.ptsrc_cat, args.flux_cut, args.out_dir / holes))
    print(f"search area {hit.sum() * hp.nside2pixarea(4096, degrees=True):.0f} deg^2")
    np.save(args.out_dir / f"footprint_hpx4096_{name}.npy", np.where(hit)[0])
    shape, wcs = enmap.read_map_geometry(str(args.obs / "websky_f150_beam_noise.fits"))
    mask = car_mask(hit, shape, wcs)
    mfile = f"surveyMask_{name}.fits"
    enmap.write_map(str(args.out_dir / mfile), enmap.enmap(mask, wcs))
    cfg = config(args.mode, mfile, boxes, args.debug, holes, args.obs)
    print(f"{len(cfg['tileDefinitions'])} nemo tiles")
    (args.out_dir / f"{name}.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {args.out_dir / mfile} and {name}.yml")


if __name__ == "__main__":
    main()
