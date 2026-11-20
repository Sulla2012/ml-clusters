# Handoff

Living status document for whoever picks this up next (human or Claude). Update it as you go.

_Last updated: 2026-11-20 (branch `refactor-centernet`)._

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
| "observed" maps: DR6 beam, beam + DR6 noise, noise only, DR6 ivar on WebSky pixels — **v1, noise explodes to 1e19 uK at DR6 ivar edges, don't use for new work** | `/mnt/welch/USERS/jorlo/ml-clusters/maps/observed/` |
| tiles from those maps (same centres as `packed_v3_cibfix`) | `websky_tiles/packed_v5_beam/`, `websky_tiles/packed_v5_beam_noise/` |
| **observed maps v2** (ivar < 1% of median → unobserved; noise ≤ 1.7e3 / 3.3e3 / 1.7e4 uK) (**use this**) | `/mnt/welch/USERS/jorlo/ml-clusters/maps/observed_v2/` |
| tiles from v2 maps (same centres) + own `split_act.npz` (**use this**) | `websky_tiles/packed_v6_beam_noise/` (built by `scripts/run_v6_data.sh`) |
| blind 4° sky patches over the test footprint (+ `footprint_hpx4096.npy`) | `websky_tiles/sky_patches_4deg_v6/` (v1 maps: `sky_patches_4deg_beam_noise/`) |
| nemo (matched filter) install, separate venv | `~/dev/nemo` (clone of simonsobs/nemo, one local patch in `nemo/pipelines.py`, see below), `~/venvs/nemo` |
| nemo runs: DR5-style search on v2 maps (**use this**), f150 source catalogue, configs | `/mnt/welch/USERS/jorlo/ml-clusters/nemo/dr5_v2/` (`WebSkyDR5/`, `WebSkyPS150/`) |
| nemo DR6-style run on v1 maps (broken by the noise bug; kept for the record) | `/mnt/welch/USERS/jorlo/ml-clusters/nemo/WebSkyACTTest/` |
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

**Against ACT DR5** (Hilton+21, arXiv:2009.11043: 90% complete for M500c^UPP > 3.8e14 at z = 0.5,
SNR > 5, ~3% false = 0.0057/deg²), from `scripts/compare_act.py` → `reports/eval/act_comparison.json`:
n2 at purity 0.97 (any matched M200m > 1e14 halo counts as real; t = 0.58, 0.034 false/deg²) is
90% complete at true M500c = 3.5e14 for 0.3 < z < 0.7; at ACT's false rate (t = 0.70) it is 3.7e14.
Roughly on par with the matched filter, but our maps are easier (DR6 depth, f220, idealised noise,
no masks) and the mass scales differ (UPP vs true). Fair benchmark still to do: nemo on these sims.

## Blind search vs nemo (done 2026-11-20)

Report §4.2. Both methods search the same sky blindly: ClusterNet on 4° patches
(`make_sky_patches.py` → `eval_sky.py --model`), nemo with the **ACT DR5 setup**
(`make_nemo_config.py --mode dr5`: 98+150 GHz, 16 UPP filters, one pass, S/N > 4, sources > 10 mJy
from an f150 nemo search (`--mode ptsrc`) masked and filled, depth cut at 10% of median). Scored by
`eval_sky.py --nemo/--rescore --footprint <nemo area> --erode_arcmin 12` (1,535 deg²; nemo makes
spurious detections along our ragged test-tile-union boundary, gone after a 12′ trim), then
`compare_act.py` → `reports/eval/sky_v6/dr5area_erode12/`.

| | 97% purity: M90 [1e14] | ACT SNR>5 false rate: M90 [1e14] |
|---|---|---|
| nemo, DR5 setup | 3.79 ± 0.09 (S/N 4.8) | 4.03 ± 0.12 (S/N 5.2) |
| n2 retrained on v6 (`clusternet-n2v6-obs`) | 3.80 ± 0.10 (t 0.68) | 3.84 ± 0.13 (t 0.73) |
| n2 (v5 maps) | 3.76 (t 0.71) | 3.87 (t 0.78) |

**ClusterNet ties the matched filter at the 90% point and nemo reproduces ACT DR5 (3.8e14).**
ClusterNet is more complete below 3.5e14 (0.47 vs 0.34 at 2.9e14). Retraining on fixed maps:
no change (test F1 0.489 vs 0.488 on v6 tiles). **Use `clusternet-n2v6-obs`.**

**Band test (2026-11-20, `reports/eval/sky_v6/bands_erode12/`):** ClusterNet on f090+f150 only
(`train.py --channels 0 1` → `clusternet-n2v6-2ch`; channels are stored in the checkpoint's
normaliser) vs nemo with f220 added (`make_nemo_config.py --bands 090 150 220`, f220 sources from
`--mode ptsrc --bands 220` masked too → `nemo/dr5_v2/WebSkyDR5f220`). Completeness at 2.2–3.4e14
(97% purity): nemo 2-band 0.268, nemo 3-band 0.289, ClusterNet 2-band 0.296, **ClusterNet 3-band
0.362** (±0.017). So without f220 ClusterNet ≈ matched filter; **its low-mass edge is f220, which
the linear matched filter can't exploit.**

nemo notes: run as `cd <dir> && mpiexec -n N ~/venvs/nemo/bin/nemo <cfg>.yml -M`
(`/usr/lib64/openmpi/bin/mpiexec`, niced; DR5 run ~45 min on 12 ranks). It crashes *after* writing
the catalogue (`KeyError: selFnOptions`, selection-function step; harmless). Local patch in
`~/dev/nemo/nemo/pipelines.py` (`addSiphonedFromSets`: skip empty catalogues before `vstack`,
newer astropy refuses to stack their float `name` column) — only needed for the DR6 multi-pass
config; could be upstreamed. The DR6-style run on v1 maps (`nemo/WebSkyACTTest`) is broken by the
noise bug below; don't use it.

**Noise bug (fixed):** v1 observed maps had white noise 1/sqrt(ivar) with DR6 edge ivar down to
5e-38 → noise up to 1e19 uK in 6,735 tiles (~1% of ACT-split tiles) and 26/241 patches, which
broke nemo's filters. `make_observed_maps.py --ivar_floor 0.01` (default) treats ivar < 1% of
median as unobserved → `maps/observed_v2`, `packed_v6_beam_noise`, `sky_patches_4deg_v6`
(`scripts/run_v6_data.sh`).

## Open issues / next steps

0. Check the f220 gain survives a different CIB model (WebSky's CIB sets how much f220 helps);
   look at what the network does with f220 (saliency near dusty sources). Run both on a contiguous footprint so no
   edge trim is needed. Train on random 2–4° crops (blind patches have higher background score
   than 1° tiles). Fresh noise per epoch (one realisation now); score against a mass- or
   S/N-limited catalogue; completeness vs local depth (per-tile depth is in `split_act.npz`).
   Compare with the existing nemo-on-WebSky catalogues in `/mnt/welch/USERS/cwhitaker/maps/websky/`
   (`NemoWebSky_..._ACT-DR5-2Pass_2degTiles_optimalCatalog.fits`).

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
- 2026-11-19/20: nemo installed (`~/dev/nemo`, `~/venvs/nemo` via uv — keeps the shared `~/.local`
  numpy/pixell untouched). DR6-style run failed (noise bug); fixed maps (v2/v6), DR5-style nemo,
  retrained n2 (`clusternet-n2v6-obs`, 20 epochs, ~45 min). Report §4.2 and §5.4. Report rebuild now
  also takes `"n2 retrained (fixed maps)"=$M/clusternet-n2v6-obs` in `--runs` and
  `--png nemo=reports/eval/sky_v6/dr5area_erode12/act_comparison.png`
  (later also `"n2, f090+f150 only"=$M/clusternet-n2v6-2ch` and
  `bands=reports/eval/sky_v6/bands_erode12/act_comparison.png`).
- 2026-11-20: band test (2-channel ClusterNet, 3-band nemo); scoring wait-loops must not `pgrep -f`
  a string that appears in their own command line (they match themselves and never exit).
