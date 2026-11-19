"""Is the model biased toward North-South? Signed errors from South's seat, against double dummy.

The BBO training deals give South the best hand almost always. If the model carries that over, it expects too many
tricks from North-South contracts and too few from East-West ones: then South's doubles look better than they are
and its own contracts safer than they are. Measured on deals where South is NOT specially strong.

  python bias_check.py --run RUN [--run RUN2 ...] [--live 3000] [--workers 8]

Deals: the Brill rows of brill_test.npz (events of 6 to 17 November), and --live of the 10,000 random deals of the
live test with the auctions of the doubles-and-sacrifices run. For each, South's view at the end of the auction:
  final contract by East-West / by North-South: the chance it makes as the model gives it, and how often it makes
  expected tricks of the final contract, model minus real
  expected par for North-South, model minus real, before any call and at the end
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

import paths

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = paths.DATA
LIVE = os.path.join(paths.BRILL, "live")
SEATS, RANKS = "NESW", "AKQJT98765432"


def positions(n_live):
    """[(set name, South's hand, dealer, vul, calls, dd tricks {strain+declarer}, par for N-S)]"""
    out = []
    z = np.load(os.path.join(DATA, "brill_test.npz"))
    for i in range(len(z["dealer"])):
        own = z["owner"][i]
        hand = ".".join("".join(RANKS[r] for r in range(13) if own[s * 13 + r] == 2) for s in range(4))
        calls = []
        for t in z["calls"][i][: z["n_calls"][i]]:
            calls.append({1: "PASS", 2: "X", 3: "XX"}.get(int(t)) or f"{(int(t) - 4) // 5 + 1}{'CDHSN'[(int(t) - 4) % 5]}")
        dd = {s + d: int(z["dd"][i][di * 5 + si]) for di, d in enumerate(SEATS) for si, s in enumerate("SHDCN")}
        out.append(("Brill events", hand, SEATS[int(z["dealer"][i])], ["None", "N_S", "E_W", "Both"][int(z["vul"][i])],
                    calls, dd, int(z["par_ns"][i])))
    deals = {(b["tournament_id"], b["board"]): b for b in map(json.loads, open(os.path.join(LIVE, "deals10k.jsonl"), encoding="utf-8"))}
    dd_all = json.load(open(os.path.join(LIVE, "10k-report", "dd.json")))
    walks = [json.loads(line) for line in open(os.path.join(LIVE, "10k", "tac-ds", "replay.jsonl"), encoding="utf-8")]
    walks.sort(key=lambda r: (r["tournament_id"], r["board"]))
    for r in walks[:: max(1, len(walks) // n_live)][:n_live]:
        b = deals[(r["tournament_id"], r["board"])]
        first = SEATS.index(b["deal"][0])
        hand = b["deal"][2:].split()[(2 - first) % 4]
        d = dd_all[f"{r['tournament_id']}|{r['board']}"]
        calls = [{"P": "PASS"}.get(c, c) for c in r["walk"]]
        out.append(("random deals", hand, b["dealer"], {"NS": "N_S", "EW": "E_W", "All": "Both"}.get(b["vul"], b["vul"]),
                    calls, d["tricks"], int(d["par"])))
    return out


def contract(calls, dealer):
    bids = [(i, c) for i, c in enumerate(calls) if c[0].isdigit()]
    if not bids:
        return None
    i_last, last = bids[-1]
    side = (SEATS.index(dealer) + i_last) % 2
    first = next(i for i, c in bids if c[1] == last[1] and (SEATS.index(dealer) + i) % 2 == side)
    return int(last[0]), last[1], SEATS[(SEATS.index(dealer) + first) % 4]


def work(job):
    run, rows = job
    try:
        import torch
    except ImportError:
        sys.path.append(os.path.join(HERE, "..", "acbl-pipeline", ".venv", "Lib", "site-packages"))
        import torch
    sys.path.insert(0, HERE)
    import infer
    torch.set_num_threads(2)
    pred = infer.Predictor(run, device="cpu")
    torch.set_num_threads(2)
    out = []
    for name, hand, dealer, vul, calls, dd, par in rows:
        start = pred.position(hand, "S", dealer, vul, [])
        end = pred.position(hand, "S", dealer, vul, calls)
        rec = {"set": name, "hcp": sum(4 - "AKQJ".index(c) for c in hand if c in "AKQJ"),
               "par0": start.par_summary()[0] - par, "par1": end.par_summary()[0] - par, "side": None}
        c = contract(calls, dealer)
        if c:
            level, strain, decl = c
            rel = (SEATS.index(decl) - 2) % 4
            p = end.dd[rel, infer.STRAINS.index(strain)]
            real = dd[strain + decl]
            rec.update(side="NS" if decl in "NS" else "EW", p_make=float(p[level + 6:].sum()), made=real >= level + 6,
                       tricks=float((p * np.arange(14)).sum()) - real)
        out.append(rec)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--live", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = positions(a.live)
    for run in a.run:
        chunks = [(run, rows[i::a.workers * 4]) for i in range(a.workers * 4)]
        with mp.Pool(a.workers) as pool:
            recs = [r for part in pool.map(work, chunks) for r in part]
        print(f"\n{os.path.basename(os.path.normpath(run))}: South's view, model against double dummy (+ = the model favours North-South)")
        print(f"{'deals':14s}{'South':>12s}{'n':>6s}{'par before any call':>21s}{'par at the end':>16s}"
              f"{'EW contracts: n':>17s}{'model makes':>13s}{'really':>8s}{'tricks':>8s}{'NS contracts: n':>17s}{'model makes':>13s}{'really':>8s}{'tricks':>8s}")
        for name in ("Brill events", "random deals"):
            for label, lo, hi in (("all hands", 0, 40), ("0-9 points", 0, 9), ("10-14", 10, 14), ("15+", 15, 40)):
                sel = [r for r in recs if r["set"] == name and lo <= r["hcp"] <= hi]
                if len(sel) < 30:
                    continue
                line = f"{name:14s}{label:>12s}{len(sel):6d}{np.mean([r['par0'] for r in sel]):+21.0f}{np.mean([r['par1'] for r in sel]):+16.0f}"
                for side in ("EW", "NS"):
                    s = [r for r in sel if r["side"] == side]
                    line += (f"{len(s):17d}{np.mean([r['p_make'] for r in s]):13.1%}{np.mean([r['made'] for r in s]):8.1%}"
                             f"{np.mean([r['tricks'] for r in s]):+8.2f}") if s else " " * 46
                print(line)
