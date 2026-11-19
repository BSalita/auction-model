"""How often would the par guard propose a call? South's passes that end the auction in the other side's partscore,
taken from a replay record; prints how many get a proposal at each margin and what the model expects.

  AUCTION_MODEL_CODE=/code python guard_count.py /var/lib/ben/brill/field.jsonl /var/lib/ben/brill/old/replay.jsonl /data/runs/v1-brill
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.environ.get("BEN_ROOT", "/ben"), "src"))
import auction_model_deals as amd  # noqa: E402

field, replay, run = sys.argv[1:4]
boards = {(b["tournament_id"], b["board"]): b for b in map(json.loads, open(field))}
VUL = {"None": [False, False], "NS": [True, False], "EW": [False, True], "All": [True, True]}
n_pass = 0
gains = []
for line in open(replay):
    r = json.loads(line)
    b = boards[(r["tournament_id"], r["board"])]
    dealer = "NESW".index(b["dealer"])
    hand = b["deal"][2:].split()[2]
    walk = r["walk"]
    if len(walk) < 4 or walk[-3:] != ["P", "P", "P"] or (dealer + len(walk) - 1) % 4 != 2:
        continue                                        # South's pass did not end the auction
    auction = ["PAD_START"] * dealer + ["PASS" if c == "P" else c for c in walk[:-1]]
    n_pass += 1
    got = amd.guard(run, hand, 2, dealer, VUL[b["vul"]], auction, -10000)
    if got:
        gains.append((got[1] - got[2], got[0], "-".join(walk[:-1]), hand))
print(f"{n_pass} boards where South's pass ended the auction; the other side in an undoubled partscore with a bid of ours "
      f"available at the 3 level or lower on {len(gains)}")
for m in (0, 30, 50, 100, 150):
    print(f"  model expects {m}+ points more from bidding than defending: {sum(g[0] >= m for g in gains)}")
for g in sorted(gains, reverse=True)[:8]:
    print(f"  {g[0]:+5.0f}  {g[1]}  after {g[2]}  holding {g[3]}")
