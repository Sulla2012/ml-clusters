# ml-clusters

Detect galaxy clusters in multi-frequency CMB maps with a convolutional neural network.

The detector (`ClusterNet`) is a ConvNeXt U-Net with CenterNet-style heads: it predicts a
full-resolution cluster-centre heatmap and a cluster radius map. Training targets are
discs/Gaussians whose size scales with each cluster's angular size θ500c, which is derived
from its mass and redshift. It is anchor-free, fully convolutional (any tile size, any
number of frequency channels), and outputs a sky catalogue with positions, scores and
sizes.

See `reports/report.html` for results and `HANDOFF.md` for project status.

## Install

```bash
uv pip install -e ".[dev]"     # or: python -m pip install --user -e ".[dev]"
```

## Pipeline

All scripts take `--help` and a `--debug` flag for quick tests.

```bash
D=/mnt/welch/USERS/jorlo/ml-clusters
WS=/mnt/welch/USERS/cwhitaker/maps/websky

# 1. Catalogue with M500c and theta500c (reads only the mass-sorted head of halos.pksc)
python scripts/build_catalog.py $WS/halos.pksc $D/catalogs/websky_m1e14_z0.05.npz

# 2a. Pack existing FITS tiles into one memmap ...
python scripts/pack_tiles.py $D/websky_tiles/packed_large $D/websky_tiles/large_tiles/*_stamps
# 2b. ... or cut new tiles from full-sky maps (all channels on one TAN grid)
python scripts/make_tiles.py $D/catalogs/websky_m1e14_z0.05.npz OUT_DIR \
    $WS/websky_f090_map.fits $WS/websky_f150_map.fits $WS/websky_f220_map.fits

# 3. Per-tile cluster lists (positions, masses, sizes) + leak-free HEALPix train/val/test split
python scripts/make_targets.py $D/websky_tiles/packed_large $D/catalogs/websky_m1e14_z0.05.npz

# 4. Train, then evaluate (purity/completeness of the de-duplicated sky catalogue)
python scripts/train.py $D/websky_tiles/packed_large $D/models/clusternet-v1
python scripts/evaluate.py $D/websky_tiles/packed_large $D/models/clusternet-v1/best.pt reports/eval_v1

# Build the HTML report from training logs and evaluation outputs
python scripts/make_report.py --help
```

## Layout

| module | contents |
|---|---|
| `mlclusters.cosmology` | WebSky cosmology, distances, densities |
| `mlclusters.halos` | M200m→M500c (NFW + Duffy08), θΔ, mask radius |
| `mlclusters.catalogs` | WebSky / nemo catalogue readers → `Catalog` |
| `mlclusters.geometry` | vectorised TAN tile WCS, sky matching, de-duplication, spatial split |
| `mlclusters.masks` | per-tile targets, mass-scaled instance masks, heatmaps, ignore regions |
| `mlclusters.data` | packed-tile storage, normalisation, `TileDataset` |
| `mlclusters.model` | `ClusterNet`, focal + radius losses, peak decoding |
| `mlclusters.training` | training loop, checkpoints |
| `mlclusters.detect` | inference, catalogue building and scoring |
| `mlclusters.stamps` | cutting tiles from full-sky maps |
| `mlclusters.websky` | WebSky sky-model map maker |
| `mlclusters.legacy` | loader for the old Faster R-CNN models (baseline only) |

Old notebooks are in `notebooks/` (see its README).
