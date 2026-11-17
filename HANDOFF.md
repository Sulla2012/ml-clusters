# Handoff

Living status document for whoever picks this up next (human or Claude). Update it as you go.

_Last updated: 2026-11-17 (branch `refactor-centernet`)._

## Where things are

| what | path |
|---|---|
| code | `mlclusters/` (library), `scripts/` (CLIs, all with `--debug`), `tests/` |
| report | `reports/report.html` (self-contained; rebuild with `scripts/make_report.py`) |
| halo catalogue (M200m > 1e14, z > 0.05, with M500c/θ500c) | `/mnt/welch/USERS/jorlo/ml-clusters/catalogs/websky_m1e14_z0.05.npz` |
| packed WebSky tiles (original, f150/f220 misaligned by 1 px) | `/mnt/welch/USERS/jorlo/ml-clusters/websky_tiles/packed_large/` |
| packed WebSky tiles, channels realigned | `/mnt/welch/USERS/jorlo/ml-clusters/websky_tiles/packed_large_aligned/` |
| corrected WebSky f150 map (CIB from cib_153) | `/mnt/welch/USERS/jorlo/ml-clusters/maps/websky_f150_map_cibfix.fits` |
| tiles re-cut from corrected maps, same centres, 121×121 (**use this**) | `/mnt/welch/USERS/jorlo/ml-clusters/websky_tiles/packed_v3_cibfix/` |
| models | `/mnt/welch/USERS/jorlo/ml-clusters/models/clusternet-*/` (`best.pt`, `log.jsonl`, `config.json`) |
| legacy Faster R-CNN | `/mnt/welch/USERS/jorlo/ml-clusters/models/torch-websky/` |

Each packed directory holds `images.npy` (N, 3, 120, 120 float32 memmap, uK), `tiles.npz` (per-tile
TAN WCS), `targets.npz` (every M200m > 1e14 halo in/near each tile, ragged), `split.npz` (spatial
HEALPix nside=4 split: 0 train / 1 val / 2 test / −1 boundary) and `split_legacy.npz` (the old
train/test directories, for comparisons with the legacy model only — it leaks sky).

## Decisions made

- **Mask size** = θ500c (from M200m via NFW + Duffy08 concentration, WebSky cosmology, d_A),
  clipped to [1′, 6′]; configurable (`--mask_scale`, `--r_min`, `--r_max`). Halos with
  1e14 ≤ M200m < 2e14, and positives centred just off-tile, are *ignore* regions in the loss.
- **Network**: ConvNeXt U-Net (2.6 M params) + CenterNet heads (centre heatmap with focal loss,
  log-radius L1). Full-resolution output because clusters are only 2–12 px across. Replaced
  Faster R-CNN (anchors, ImageNet normalisation, 120→800 px upsampling, no ignore regions).
- **Inputs**: per-tile median removed, divided by a global per-channel robust σ, then
  `3·asinh(x/3)`; stored in the checkpoint.
- **Evaluation**: detections within 2 px of a tile edge dropped, sky-NMS at 2′, greedy one-to-one
  matching at 2′; completeness on truths inside the footprint; threshold chosen on val (max F1).

## Open issues / next steps

1. ~~Regenerate the WebSky f150 map~~ Done 2026-11-17: the original (cwhitaker's dir, left untouched)
   used the 90 GHz CIB map. The new map differs from it by exactly 2594.7·(cib_153 − cib_90.2)
   (0.000 uK residual), ~22 uK rms. Tiles re-cut from it into `packed_v3_cibfix`.
2. **Domain gap to ACT**: the WebSky maps have no beam and no noise. Add per-band beam
   convolution and realistic (ideally ACT DR6 ivar-based) noise before applying to real data.
3. Try a mass/SNR-dependent positive cut (many 2e14 halos at high z are below the noise in real
   data); consider regressing log M500c as an extra head.
4. Run inference on large tiles / full maps (the network is fully convolutional) instead of
   per-cluster 1° tiles, then evaluate against the full halo catalogue.
5. Fix the broken venv (`~/venvs/general/bin/python` resolves to the system interpreter, so the
   venv's site-packages are never used; everything runs from `~/.local`). The package was
   installed with `python3.12 -m pip install --user -e .` for that reason.

## Log

- 2026-11-17: refactor into `mlclusters`; mass-scaled masks; ClusterNet; bug audit (see report).
- 2026-11-17: regenerated f150 map with CIB fix (5 min, 23 GB RAM); re-cut tiles; training v3 on them.
