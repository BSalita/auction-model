#!/bin/bash
# The base model again on BBO rows of all four dealers (bbo4_*.npz = the old files plus the North-dealer boards
# of extract_bbo_lin.py), then the v1-brill recipe on top of it.
#   bash ft4.sh            (inside the container; about an hour and a quarter on a free GPU)
#   v3         v1's recipe (3 passes, batch 768, 4e-4) on bbo4_train.npz
#   v3-brill   v3 + 2 passes at 3e-5 over brill_train.npz, like v1-brill
# Then the held-out tables of v1, v1-brill, v3, v3-brill on: the North-dealer BBO deals (bbo_north_test.npz),
# all BBO held-out deals of the old kind (bbo_heldout.npz), the Brill rows of 6 to 17 November (brill_test.npz)
# and of 18 November (brill_new.npz).
export PYTHONPATH=/var/lib/ben/torch-pkgs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /code
R=/data/runs
mkdir -p $R/v3 $R/v3-brill
python train.py --data /data --run $R/v3 --epochs 3 --train-file bbo4_train.npz --test-file bbo4_test.npz > $R/v3/stdout.txt 2>&1
echo "v3 trained $(date -u +%H:%M:%S)" >> $R/v3/ft4.log
python train.py --data /data --init $R/v3/model.pt --run $R/v3-brill --eval-rows 0 --log-every 20 --batch 128 --warmup 20 \
    --lr 3e-5 --epochs 2 --train-file brill_train.npz --test-file brill_test.npz > $R/v3-brill/stdout.txt 2>&1
echo "v3-brill trained $(date -u +%H:%M:%S)" >> $R/v3/ft4.log
for m in v1 v1-brill v3 v3-brill; do
  for t in bbo_north_test brill_new brill_test bbo_heldout; do
    rows=""; case $t in brill_*) rows="--eval-rows 0";; esac
    python train.py --data /data --run $R/$m --eval-only --test-file $t.npz $rows > $R/$m/x_$t.txt 2>&1
  done
  echo "$m checked $(date -u +%H:%M:%S)" >> $R/v3/ft4.log
done
echo "done $(date -u +%H:%M:%S)" >> $R/v3/ft4.log
