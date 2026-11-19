#!/bin/bash
# Clean-vs-observed training experiment, end to end: DR6 noise fit, beam/noise maps, tiles,
# targets, ACT-footprint split, then train and evaluate five models (see HANDOFF.md).
# Usage: scripts/run_noise_experiment.sh   (from the repo root; ~6 h on one GPU)
set -x
cd /home/jorlo/dev/ml-clusters
D=/mnt/welch/USERS/jorlo/ml-clusters
O=$D/maps/observed
T=$D/websky_tiles
M=$D/models
CAT=$D/catalogs/websky_m1e14_z0.05.npz
# 1. measure the DR6 noise shape and make the observed maps
python3.12 scripts/fit_act_noise.py $D/maps > $D/maps/fit_act_noise.log 2>&1 || { echo "NOISE_FIT_FAILED"; exit 1; }
python3.12 scripts/make_observed_maps.py $O > $O/make.log 2>&1 || { echo "MAPS_FAILED"; exit 1; }
# 2. tiles at the packed_v3_cibfix centres, targets, footprint split
for kind in beam beam_noise; do
  out=$T/packed_v5_$kind
  python3.12 scripts/make_tiles.py $CAT $out $O/websky_f090_$kind.fits $O/websky_f150_$kind.fits $O/websky_f220_$kind.fits --centres_from $T/packed_v3_cibfix -n 4 > $out.log 2>&1 || { echo "TILES_FAILED $kind"; exit 1; }
  python3.12 scripts/make_targets.py $out $CAT >> $out.log 2>&1 || { echo "TARGETS_FAILED $kind"; exit 1; }
done
python3.12 scripts/footprint_split.py $T/packed_v3_cibfix $T/packed_v5_beam $T/packed_v5_beam_noise --ivar $O/act_dr6_f090_ivar.fits $O/act_dr6_f150_ivar.fits $O/act_dr6_f220_ivar.fits > $T/split_act.log 2>&1 || { echo "SPLIT_FAILED"; exit 1; }
echo TILES_DONE
# 3. train and evaluate (needs the GPU to be free)
ev() {  # ev <model_dir> <tiles_dir> <name>
  python3.12 scripts/evaluate.py $2 $1/best.pt reports/eval/noise/$3 --split split_act.npz --workers 4 > reports/eval/noise/$3.log 2>&1
}
mkdir -p reports/eval/noise
train() {  # train <tiles_dir> <model_name> [extra args]
  local t=$1 n=$2; shift 2
  python3.12 scripts/train.py $t $M/$n --split split_act.npz --workers 7 "$@" > $M/$n.log 2>&1
}
train $T/packed_v5_beam_noise clusternet-n2-obs --epochs 20
train $T/packed_v3_cibfix clusternet-n0-clean-fp --epochs 20
for m in clusternet-v3-cibfix clusternet-n0-clean-fp clusternet-n2-obs; do
  for d in v3_cibfix v5_beam v5_beam_noise; do ev $M/$m $T/packed_$d ${m#clusternet-}_on_$d; done
done
train $T/packed_v5_beam clusternet-n1-beam --epochs 20
train $T/packed_v5_beam_noise clusternet-n3-obs-ft --epochs 8 --lr 5e-4 --warmup 0.2 --init $M/clusternet-v3-cibfix/best.pt
for m in clusternet-n1-beam clusternet-n3-obs-ft; do
  for d in v3_cibfix v5_beam v5_beam_noise; do ev $M/$m $T/packed_$d ${m#clusternet-}_on_$d; done
done
echo NOISE_ALL_DONE
