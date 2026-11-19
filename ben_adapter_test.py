"""Smoke test and timing of BEN bidding with its deals taken from the auction model.

  python ben_adapter_test.py --conf A.conf --conf-model B.conf [--data /data/brill_test.npz --rows 20]

A is the ordinary config, B the same with `[sampling] auction_model = <run>`. For South's turns of the first rows
it prints both calls, the seconds each took, and the time the model spent drawing the deals.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.environ.get("BEN_ROOT", "/ben"), "scripts", "match"))
import ab_match  # noqa: E402

from ben_samples import RANKS, call_str  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--conf", required=True)
ap.add_argument("--conf-model", required=True)
ap.add_argument("--data", default="/data/brill_test.npz")
ap.add_argument("--rows", type=int, default=20)
a = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.chdir(ab_match.SRC)
from nn.models_tf2 import Models  # noqa: E402
from ddsolver.ddsolver import DDSolver  # noqa: E402
from botbidder import BotBid  # noqa: E402
from sample import Sample  # noqa: E402
import auction_model_deals  # noqa: E402

confs = [ab_match.load_conf(ab_match.resolve(p), None) for p in (a.conf, a.conf_model)]
models = [Models.from_conf(c, ab_match.BEN_ROOT) for c in confs]
for m in models:
    ab_match.platform_fixups(m)
dds = DDSolver(max_threads=1)
D = np.load(a.data)
run = Sample.from_conf(confs[1]).auction_model
print("auction model run:", run)
same = total = 0
secs = [[], []]
draw = []
for row in range(a.rows):
    owner, dealer, vul, n = D["owner"][row], int(D["dealer"][row]), int(D["vul"][row]), int(D["n_calls"][row])
    calls = [call_str(t) for t in D["calls"][row, :n]]
    hand = ".".join("".join(RANKS[r] for r in range(13) if owner[s * 13 + r] == 2) for s in range(4))
    vuln = [bool(vul & 1), bool(vul & 2)]
    for t in range(n):
        if (dealer + t) % 4 != 2:
            continue
        auction = ["PAD_START"] * dealer + calls[:t]
        got = []
        for k in (0, 1):
            np.random.seed(42)
            bot = BotBid(vuln, hand, models[k], Sample.from_conf(confs[k]), 2, dealer, dds, False, False)
            t0 = time.time()
            got.append(bot.bid(auction).bid)
            secs[k].append(time.time() - t0)
        s = 0.0
        if run:
            _, hcp, shp, s = auction_model_deals.deals(run, hand, 2, dealer, vuln, auction, 50, np.random.default_rng(1))
        draw.append(s)
        total += 1
        same += got[0] == got[1]
        print(f"row {row:>3} {hand:<18} after {' '.join(calls[:t]) or '-':<40} real {calls[t]:<5} BEN {got[0]:<5} "
              f"with model {got[1]:<5} {secs[0][-1]:.2f}s {secs[1][-1]:.2f}s  draw {s:.3f}s", flush=True)
print(f"{total} turns, same call {same}; seconds a call: ordinary mean {np.mean(secs[0]):.2f} median {np.median(secs[0]):.2f}, "
      f"with model mean {np.mean(secs[1]):.2f} median {np.median(secs[1]):.2f}; drawing 50 deals: mean {np.mean(draw):.3f}s "
      f"median {np.median(draw):.3f}s (first call includes loading)")
