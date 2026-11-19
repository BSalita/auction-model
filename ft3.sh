#!/bin/bash
# The v1-brill recipe (v1 + 2 passes at 3e-5, batch 128) on more Brill rows.
#   bash ft3.sh            (inside the container, after extract_brill.py has rebuilt brill.npz)
# Needs in /data: brill.npz (all rows), brill_old.npz (the rows v1-brill's recipe was chosen on, train + test),
# brill_new.npz (events after them: rows no earlier model has seen), bbo_heldout.npz (a copy of bbo_test.npz, so
# that an eval-only run does not overwrite a run's eval.txt).
#   v1-brill-old   trained on brill_old.npz: what the newer events say about adding the 12 held-out days
#   v1-brill-all   trained on brill.npz: the model for use; nothing is held out from it
export PYTHONPATH=/var/lib/ben/torch-pkgs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /code
R=/data/runs
T="--data /data --init $R/v1/model.pt --eval-rows 0 --log-every 20 --batch 128 --warmup 20 --lr 3e-5 --epochs 2"
run() { name=$1; shift; rm -rf $R/$name; mkdir -p $R/$name; python train.py $T --run $R/$name "$@" > $R/$name/stdout.txt 2>&1; echo "$name trained $(date -u +%H:%M:%S)"; }
run v1-brill-old --train-file brill_old.npz --test-file brill_new.npz
run v1-brill-all --train-file brill.npz --test-file brill_new.npz
for m in v1 v1-brill v1-brill-old v1-brill-all; do
  python train.py --data /data --run $R/$m --eval-only --test-file brill_new.npz --eval-rows 0 > $R/$m/x_brill_new.txt 2>&1
  python train.py --data /data --run $R/$m --eval-only --test-file bbo_heldout.npz > $R/$m/x_bbo_heldout.txt 2>&1
  echo "$m checked $(date -u +%H:%M:%S)"
done
