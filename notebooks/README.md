# Historical notebooks

Exploratory notebooks from before the `mlclusters` refactor. They import modules that
no longer exist (`stamping_utils`, `utils.nn`, `visionutils`, `constants`); to run them,
check out commit `6fea0a3` (last pre-refactor commit on `main`). The FITS masks and
`data.h5` live here because the notebooks reference them by relative path.

Their functionality now lives in:

| notebook | replacement |
|---|---|
| `halo-masks.ipynb` (mass-scaled masks WIP) | `mlclusters.halos`, `mlclusters.masks`, `scripts/make_targets.py` |
| `stamper_full.ipynb`, `HaloFile-2.ipynb` | `mlclusters.stamps`, `scripts/make_tiles.py` |
| `websky_map_maker.ipynb` | `mlclusters.websky`, `scripts/make_websky_map.py` |
| `torch-mask-rcnn.ipynb` | `mlclusters.model`, `scripts/train.py`, `scripts/evaluate.py` |
| `torch-classifier.ipynb` | (backbone pre-training is no longer needed) |
