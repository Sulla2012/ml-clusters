# Handoff

Living status document for whoever picks this up next (human or Claude). Update it as you go.

_Last updated: 2026-11-19 (branch `refactor-centernet`)._

## Where things are

| what | path |
|---|---|
| code | `mlclusters/` (library), `scripts/` (CLIs, all with `--debug`), `tests/` |
| report | `reports/report.html` (self-contained; rebuild with `scripts/make_report.py`, command below) |
| explainer page (claude.ai artifact, private) | https://claude.ai/artifact/KTy7Hd7GtVDGrTTKB1qsDU |
| halo catalogue (M200m > 1e14, z > 0.05, with M500c/θ500c) | `/mnt/welch/USERS/jorlo/ml-clusters/catalogs/websky_m1e14_z0.05.npz` |
| packed WebSky tiles (original, f150/f220 misaligned by 1 px) | `/mnt/welch/USERS/jorlo/ml-clusters/websky_tiles/packed_large/` |
| packed WebSky tiles, channels realigned | `/mnt/welch/USERS/jorlo/ml-clusters/websky_tiles/packed_large_aligned/` |
| corrected WebSky f150 map (CIB from cib_153) | `/mnt/welch/USERS/jorlo/ml-clusters/maps/websky_f150_map_cibfix.fits` |
| tiles re-cut from corrected maps, same centres, 121×121 (**use this**) | `/mnt/welch/USERS/jorlo/ml-clusters/websky_tiles/packed_v3_cibfix/` |
| ACT DR6 noise shapes (measured) | `/mnt/welch/USERS/jorlo/ml-clusters/maps/act_noise_f{090,150,220}.json` (+ `.npz` spectra) |
| "observed" maps: DR6 beam, beam + DR6 noise, noise only, DR6 ivar on WebSky pixels | `/mnt/welch/USERS/jorlo/ml-clusters/maps/observed/` |
| tiles from those maps (same centres as `packed_v3_cibfix`) | `websky_tiles/packed_v5_beam/`, `websky_tiles/packed_v5_beam_noise/` |
| ACT-footprint split (in v3_cibfix, v5_beam, v5_beam_noise) | `split_act.npz` (`split`, per-tile `depth` [uK arcmin]) |
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

- **Beam and noise** (`mlclusters/instrument.py`, `scripts/fit_act_noise.py`,
  `scripts/make_observed_maps.py`): clean maps are convolved with the ACT DR6 coadd beam
  (`/mnt/welch/ACT/20240323_simple/beam_f*_tform.txt`, lmax 21600). Noise = white with the DR6
  daynight T ivar (ACT-only, s08–s22; s17–s22 at f220) + an atmospheric part scaled to the local depth, whose
  spectrum relative to white is *measured* from DR6 daynight − night srcfree differences in 7–8
  10° patches (tabulated; a power-law fit gives ell_knee ≈ 2000 / 2600 / 4600 and alpha ≈
  −3.8 / −4.4 / −3.5 at f090 / f150 / f220; deep-patch depth ≈ 9 / 9 / 48 uK arcmin). One fixed
  realisation per band, independent between bands. Footprint = DR6 ivar > 0 (48.8% of sky).

## Noise experiment (done 2026-11-18)

Question: is it better to train on clean (no beam, no noise) or on observed (beam + noise) maps?
All models are evaluated on the ACT-footprint test tiles (`--split split_act.npz`) of clean,
beam-only and beam+noise tiles (`reports/eval/noise/<model>_on_<data>`):
v3 (clean, all sky), n0 (clean, footprint only: same tiles as the others), n1 (beam only),
n2 (beam + noise), n3 (v3 fine-tuned 8 epochs on beam + noise). The whole pipeline is in
`scripts/run_noise_experiment.sh`.

Test F1 (rows: trained on; cols: tested on), threshold chosen per pair on val:

| | clean | beam | beam + noise |
|---|---|---|---|
| v3 clean, all sky | **0.876** | 0.570 | 0.054 |
| n0 clean, footprint | 0.868 | 0.552 | 0.053 |
| n1 beam | 0.314 | **0.852** | 0.058 |
| n2 beam + noise | 0.785 | 0.848 | **0.486** |
| n3 v3 → ft beam + noise | 0.780 | 0.828 | 0.469 |

**Answer: train on beam + noise (n2).** Clean-trained models fire on noise everywhere (purity
2–4% at any threshold). n2 is ~complete above 6e14 (0.98–0.99), 0.83 at 4–6e14, 0.27 at
2–3e14; train loss = val loss and val F1 plateaus after ~4 epochs, so the low F1 is
detectability (many 2e14 targets are below the noise), not overfitting. Fine-tuning from v3
gives the same P–C curve as training from scratch. **Use n2 as the starting point for ACT-like work.**

## Open issues / next steps

0. Noise follow-ups: fresh noise per epoch (only one realisation now, model may memorise it);
   score against a mass- or S/N-limited catalogue; completeness vs local depth (per-tile depth is in
   `split_act.npz`).

1. ~~Regenerate the WebSky f150 map~~ Done 2026-11-17: the original (cwhitaker's dir, left untouched)
   used the 90 GHz CIB map. The new map differs from it by exactly 2594.7·(cib_153 − cib_90.2)
   (0.000 uK residual), ~22 uK rms. Tiles re-cut from it into `packed_v3_cibfix`.
2. **Domain gap to ACT**: beam + DR6-like noise maps now exist (see above). Still missing:
   noise correlated between bands, anisotropic (scan-direction) noise, map-maker transfer
   function, point-source masking/subtraction as done on real data. Fresh noise per epoch
   (on-the-fly in the data loader) would avoid training on a single noise realisation.
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
- 2026-11-18: overnight queue died with the session; relaunched detached (v3, v4, legacy split). Measured DR6 noise; built beam/noise maps; queued the clean-vs-noisy training experiment.
- 2026-11-19: noise experiment finished overnight (n0–n3 + 15 evals). Report §4 and explainer
  artifact updated. Report rebuild:
  `python3.12 scripts/make_report.py --tiles $T/packed_large --aligned $T/packed_large_aligned
  --catalog $W/catalogs/websky_m1e14_z0.05.npz --old_masks $T/large_tiles/train_large_indv_freq_masks
  --runs v1=$M/clusternet-v1 v2=$M/clusternet-v2-aligned v3=$M/clusternet-v3-cibfix v4=$M/clusternet-v4-mass
  "n2 (beam+noise)"=$M/clusternet-n2-obs --evals v1=reports/eval/v1 ... v4=reports/eval/v4
  --legacy_evals "legacy FRCNN"=reports/eval/legacy_frcnn_legacysplit ClusterNet=reports/eval/clusternet_legacysplit
  --example_ckpt $M/clusternet-v3-cibfix/best.pt --example_tiles $T/packed_v3_cibfix --radius_eval reports/eval/v3
  --mass_eval reports/eval/v4 --noise_dir $W/maps --noise_evals reports/eval/noise
  --obs_tiles clean=$T/packed_v3_cibfix beam=$T/packed_v5_beam "beam + noise"=$T/packed_v5_beam_noise`
  (W=/mnt/welch/USERS/jorlo/ml-clusters, T=$W/websky_tiles, M=$W/models).
