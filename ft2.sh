#!/bin/bash
# Second round: short, fully annealed Brill-only runs (the first round overfitted after one pass).
export PYTHONPATH=/var/lib/ben/torch-pkgs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /code
B=/data/runs/v1
T="--data /data --init $B/model.pt --test-file brill_test.npz --eval-rows 0 --log-every 50 --train-file brill_train.npz --batch 128 --warmup 20"
run() { name=$1; shift; rm -rf /data/runs/$name; mkdir -p /data/runs/$name; python train.py $T --run /data/runs/$name "$@" > /data/runs/$name/stdout.txt 2>&1; echo "#### $name"; grep -A7 "^South" /data/runs/$name/eval.txt | cut -c1-140; }
run ft-a --lr 3e-5 --epochs 1
run ft-b --lr 3e-5 --epochs 2
run ft-c --lr 1e-5 --epochs 3
run ft-d --lr 6e-5 --epochs 1
