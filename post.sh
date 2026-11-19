#!/bin/bash
# After a training run: wait for the trainer (PID $2) to exit, then run the held-out checks.
#   bash post.sh /data/runs/v1 PID
RUN=$1
PID=$2
export PYTHONPATH=/var/lib/ben/torch-pkgs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /code
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
{
  echo "trainer exited $(date -u +%H:%M:%S)"
  python train.py --data /data --run "$RUN" --eval-only --test-file brill.npz > "$RUN/post_brill_eval.txt" 2>&1
  python eval_sets.py --run "$RUN" --states 4000 --samples 256 > "$RUN/post_sets_bbo.txt" 2>&1
  python eval_sets.py --run "$RUN" --states 4000 --samples 256 --south-only > "$RUN/post_sets_bbo_south.txt" 2>&1
  python eval_sets.py --run "$RUN" --test-file brill.npz --states 4000 --samples 256 --south-only > "$RUN/post_sets_brill_south.txt" 2>&1
  echo "checks done $(date -u +%H:%M:%S)"
} > "$RUN/post.log" 2>&1
touch "$RUN/post.done"
