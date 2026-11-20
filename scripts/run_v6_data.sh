#!/bin/bash
# Rebuild the beam + noise data on the ivar-floored maps (maps/observed_v2): 1 deg tiles at the
# packed_v3_cibfix centres, targets, ACT-footprint split, and the 4 deg blind sky patches.
# Usage: scripts/run_v6_data.sh   (from the repo root; CPU only, ~1 h)
set -x
cd /home/jorlo/dev/ml-clusters
D=/mnt/welch/USERS/jorlo/ml-clusters
O=$D/maps/observed_v2
T=$D/websky_tiles
CAT=$D/catalogs/websky_m1e14_z0.05.npz
out=$T/packed_v6_beam_noise
maps="$O/websky_f090_beam_noise.fits $O/websky_f150_beam_noise.fits $O/websky_f220_beam_noise.fits"
nice -n 10 python3.12 scripts/make_tiles.py $CAT $out $maps --centres_from $T/packed_v3_cibfix -n 6 > $out.log 2>&1 || { echo TILES_FAILED; exit 1; }
nice -n 10 python3.12 scripts/make_targets.py $out $CAT >> $out.log 2>&1 || { echo TARGETS_FAILED; exit 1; }
cp $T/packed_v5_beam_noise/split.npz $out/split.npz
nice -n 10 python3.12 scripts/footprint_split.py $out --ivar $O/act_dr6_f090_ivar.fits $O/act_dr6_f150_ivar.fits $O/act_dr6_f220_ivar.fits >> $out.log 2>&1 || { echo SPLIT_FAILED; exit 1; }
nice -n 10 python3.12 scripts/make_sky_patches.py $out $T/sky_patches_4deg_v6 --maps $maps --workers 6 > $T/sky_patches_4deg_v6.log 2>&1 || { echo PATCHES_FAILED; exit 1; }
echo V6_DATA_DONE
