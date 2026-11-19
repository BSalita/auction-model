"""BEN's own sampled deals at every South turn of held-out auctions, for comparison with the auction model.

Runs in a BEN container (TensorFlow, src\\ben mounted at /ben); no PyTorch needed:

  python ben_samples.py --conf /var/lib/ben/ab/conf/old.conf --data /data/brill_test.npz --out /data/ben_samples_old.npz

For each state (row, calls seen, South to call) it builds BotBid as the Brill replay does and calls
sample_hands_for_auction, the function bid() takes its deals from. Saved per state, padded to --max-samples:
  row, t, n       row of the data file, calls seen, deals returned
  quality, secs   BEN's sample quality and the sampling time
  lens  (S,4,3)   suit lengths (S,H,D,C) of LHO, partner, RHO in each deal
  hon   (S,16)    seat of each honor SA..CJ: 0 South, 1 LHO, 2 partner, 3 RHO
  tricks (S,)     double-dummy tricks of the declarer of the real final contract on each deal (-1 passed out),
                  through BotBid.expected_tricks_dd, spot cards dealt as BEN deals them
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.environ.get("BEN_ROOT", "/ben"), "scripts", "match"))
import ab_match  # noqa: E402  (puts BEN's src first on sys.path)

RANKS = "AKQJT98765432"
W = {}


def call_str(t):
    if t < 4:
        return {1: "PASS", 2: "X", 3: "XX"}[int(t)]
    return f"{(t - 4) // 5 + 1}{'CDHSN'[(t - 4) % 5]}"


def worker_init(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "2"
    os.chdir(ab_match.SRC)
    log = open(os.path.join(cfg["logs"], f"worker-{os.getpid()}.log"), "a")
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    from nn.models_tf2 import Models
    from ddsolver.ddsolver import DDSolver
    W["conf"] = ab_match.load_conf(cfg["conf"], None)
    W["models"] = Models.from_conf(W["conf"], ab_match.BEN_ROOT)
    ab_match.platform_fixups(W["models"])
    W["dds"] = DDSolver(max_threads=1)
    W["S"] = cfg["max_samples"]
    z = np.load(cfg["data"])
    W["D"] = {k: z[k] for k in ("owner", "dealer", "vul", "calls", "n_calls")}


def do_row(row):
    try:
        return row, states_of(row), None
    except Exception:
        return row, [], traceback.format_exc()


def states_of(row):
    from botbidder import BotBid
    from bidding import bidding
    from sample import Sample
    D, S = W["D"], W["S"]
    owner, dealer, vul, n = D["owner"][row], int(D["dealer"][row]), int(D["vul"][row]), int(D["n_calls"][row])
    calls = [call_str(t) for t in D["calls"][row, :n]]
    hand = ".".join("".join(RANKS[r] for r in range(13) if owner[s * 13 + r] == 2) for s in range(4))
    vuln = [bool(vul & 1), bool(vul & 2)]
    full = ["PAD_START"] * dealer + calls
    out = []
    for t in range(n):
        if (dealer + t) % 4 != 2:
            continue
        np.random.seed(42)
        bot = BotBid(vuln, hand, W["models"], Sample.from_conf(W["conf"]), 2, dealer, W["dds"], False, False)
        t0 = time.time()
        hands_np, _, _, _, quality = bot.sample_hands_for_auction(full[: dealer + t], 2)
        secs = time.time() - t0
        k = min(hands_np.shape[0], S)
        lens = np.zeros((S, 4, 3), np.int8)
        hon = np.zeros((S, 16), np.int8)
        tricks = np.full(S, -1, np.int8)
        if k:
            for i in (1, 2, 3):
                x = hands_np[:k, (2 + i) % 4].reshape(k, 4, -1)
                lens[:k, :, i - 1] = x.sum(-1)
                hon[:k] += (x[:, :, :4].reshape(k, 16) > 0) * i
            auctions_np = np.ones((k, 64), np.int32) * bidding.BID2ID["PAD_END"]
            for j, c in enumerate(full):
                auctions_np[:, j] = bidding.BID2ID[c]
            pbn = bot.translate_hands(hands_np[:k], hand, k)
            contracts, soft = bot.expected_tricks_dd(pbn, auctions_np)
            if contracts[0].upper() != "PASS":
                tricks[:k] = soft.argmax(1)
        out.append((row, t, k, float(quality), secs, lens, hon, tricks))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="first N rows only")
    ap.add_argument("--max-samples", type=int, default=50)
    a = ap.parse_args()
    logs = a.out + ".logs"
    os.makedirs(logs, exist_ok=True)
    n_rows = len(np.load(a.data)["dealer"])
    rows = list(range(min(n_rows, a.limit) if a.limit else n_rows))
    cfg = {"conf": ab_match.resolve(a.conf), "data": a.data, "logs": logs, "max_samples": a.max_samples}
    t0 = time.time()
    res, errors = [], 0
    with mp.get_context("spawn").Pool(a.workers, worker_init, (cfg,)) as pool:
        for i, (row, states, err) in enumerate(pool.imap_unordered(do_row, rows, chunksize=4)):
            res += states
            if err:
                errors += 1
                open(os.path.join(logs, "errors.txt"), "a").write(f"row {row}\n{err}\n")
            if (i + 1) % 100 == 0:
                print(f"{i + 1} of {len(rows)} rows, {len(res)} states, {errors} failed rows, {time.time() - t0:.0f}s", flush=True)
    res.sort(key=lambda r: (r[0], r[1]))
    np.savez(a.out, row=np.array([r[0] for r in res], np.int32), t=np.array([r[1] for r in res], np.int16),
             n=np.array([r[2] for r in res], np.int16), quality=np.array([r[3] for r in res], np.float32),
             secs=np.array([r[4] for r in res], np.float32), lens=np.stack([r[5] for r in res]),
             hon=np.stack([r[6] for r in res]), tricks=np.stack([r[7] for r in res]))
    n = np.array([r[2] for r in res])
    print(f"done: {len(res)} states from {len(rows)} rows, {errors} failed rows, {time.time() - t0:.0f}s; "
          f"deals per state mean {n.mean():.1f}, under 15: {(n < 15).mean():.1%}, none: {(n == 0).mean():.1%}; "
          f"sampling {np.mean([r[4] for r in res]):.2f}s a state", flush=True)


if __name__ == "__main__":
    main()
