"""Brill daily events -> the same arrays as extract_bbo.py (one row per distinct deal + auction).

South is the entrant (a robot or a human); North, East and West are the Brill robots.
Reads src\\data\\brill-replay\\field.jsonl and board-table\\dd.json, writes E:\\bridge\\data\\auction-model\\brill.npz
with the extra arrays `date` (yyyymmdd) and `count` (tables with that auction).
"""
import json
import os
from collections import Counter

import numpy as np

from extract_bbo import CALL_TOKEN, MAX_CALLS, SEATS, parse_auction, parse_pbn

SRC = r"D:\OneDrive\sw\bridge\ML-Contract-Bridge\src\data\brill-replay"
OUT = r"E:\bridge\data\auction-model\brill.npz"
VUL = {"None": 0, "NS": 1, "N_S": 1, "EW": 2, "E_W": 2, "All": 3, "Both": 3}
CALL = {"P": "p", "X": "d", "XX": "r", "PASS": "p"}

dd_all = json.load(open(os.path.join(SRC, "board-table", "dd.json")))
rows = {k: [] for k in ("owner", "dealer", "vul", "calls", "n_calls", "dd", "par_ns", "count", "date")}
skipped = Counter()
for line in open(os.path.join(SRC, "field.jsonl"), encoding="utf-8"):
    b = json.loads(line)
    dd = dd_all.get(f"{b['deal']}|{b['dealer']}|{b['vul']}")
    owner = parse_pbn(b["deal"])
    if dd is None or owner is None or b["vul"] not in VUL:
        skipped["board"] += 1
        continue
    tricks = [dd["tricks"][s + d] for d in SEATS for s in "SHDCN"]
    auctions = Counter(" ".join(CALL.get(c.upper(), c.upper()) for c in t["auction"]) for t in b["tables"] if t.get("auction"))
    for a, n in auctions.items():
        toks = parse_auction(a)
        if toks is None:
            skipped["auction"] += 1
            continue
        calls = np.zeros(MAX_CALLS, np.uint8)
        calls[: len(toks)] = toks
        rows["owner"].append(owner)
        rows["dealer"].append(SEATS.index(b["dealer"]))
        rows["vul"].append(VUL[b["vul"]])
        rows["calls"].append(calls)
        rows["n_calls"].append(len(toks))
        rows["dd"].append(tricks)
        rows["par_ns"].append(dd["par_score"])
        rows["count"].append(n)
        rows["date"].append(int(b["date"].replace("-", "")))
dt = {"owner": np.uint8, "dealer": np.uint8, "vul": np.uint8, "calls": np.uint8, "n_calls": np.uint8, "dd": np.uint8,
      "par_ns": np.int16, "count": np.uint32, "date": np.int32}
np.savez(OUT, **{k: np.array(v, dtype=dt[k]) for k, v in rows.items()})
n = len(rows["dealer"])
print(f"{n:,} distinct deal+auction rows, tables {sum(rows['count']):,}, skipped {dict(skipped)}, "
      f"mean calls {np.mean(rows['n_calls']):.2f}, dates {min(rows['date'])}..{max(rows['date'])}")
