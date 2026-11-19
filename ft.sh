#!/bin/bash
# Fine-tune the BBO model on the Brill deals; every run is checked on brill_test.npz (the newest 12 days).
#   bash ft.sh            (inside the container; needs split_brill.py to have run)
export PYTHONPATH=/var/lib/ben/torch-pkgs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /code
B=/data/runs/v1
T="--data /data --init $B/model.pt --test-file brill_test.npz --eval-rows 0 --log-every 50"
python train.py --data /data --run $B --eval-only --test-file brill_test.npz > $B/x_brill_test.txt 2>&1
run() { name=$1; shift; rm -rf /data/runs/$name; mkdir -p /data/runs/$name; python train.py $T --run /data/runs/$name "$@" > /data/runs/$name/stdout.txt 2>&1; echo "$name done $(date -u +%H:%M:%S)"; }
run ft-brill      --train-file brill_train.npz --batch 128 --lr 1e-4 --warmup 30 --epochs 12
run ft-brill-lo   --train-file brill_train.npz --batch 128 --lr 3e-5 --warmup 30 --epochs 12
run ft-brill-drop --train-file brill_train.npz --batch 128 --lr 1e-4 --warmup 30 --epochs 12 --drop 0.1
run ft-mix        --train-file mix_train.npz   --batch 512 --lr 1e-4 --warmup 50 --epochs 3
