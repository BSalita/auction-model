"""Which purpose rule changed the auction, and what it was worth: a replay bid with the rules of tactical.py as
extra candidates against the baseline replay on the same deals (brill_live.py records), double dummy.

  python tactical_live_report.py --deals deals10k.jsonl --dd dd.json --base base/replay.jsonl --new tac/replay.jsonl \
      --turns tac-turns.jsonl --rules double,sacrifice [--out report.txt]

--turns is `tactical_count.py turns` run on the NEW replay with the baseline config: at every South turn the
calls the rules name and the candidates BEN has without them. A changed board is put down to the rule that
named South's first different call ("other" when no rule named it: the extra candidates also shift the
comparison between BEN's own). Needs numpy; scoring.py and tactical.py of this folder.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import STRAINS, contract_score  # noqa: E402
from tactical import SEATS, Auction  # noqa: E402
from tactical_count import VUL, imps  # noqa: E402


WIRED = ["double", "sacrifice", "interfere", "lead"]      # default for --rules: the rules the new config had, in its order


def result(walk, deal, dd):
    """Final contract as text and its double-dummy score for North-South."""
    dealer = SEATS.index(deal["dealer"])
    a = Auction(walk, dealer, (dealer + len(walk)) % 4)
    if a.last is None:
        return "passed out", 0
    _, level, strain, who = a.last
    decl = a.declarer(who, strain)
    tricks = dd["tricks"][strain + SEATS[decl]]
    sc = contract_score(level, STRAINS.index(strain), a.dbl, VUL[deal["vul"]][decl % 2], tricks)
    return f"{level}{strain}{'x' * a.dbl}{SEATS[decl]}", int(sc if decl % 2 == 0 else -sc)


def ci(x):
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    for k in ("deals", "dd", "base", "new", "turns"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--rules", default=",".join(WIRED), help="the new config's auction_model_tactical value")
    ap.add_argument("--out")
    ap.add_argument("--examples", type=int, default=5)
    a = ap.parse_args()
    key = lambda r: (r["tournament_id"], r["board"])
    deals = {key(b): b for b in map(json.loads, open(a.deals))}
    dd = json.loads(open(a.dd).read())
    base = {key(r): r for r in map(json.loads, open(a.base))}
    new = {key(r): r for r in map(json.loads, open(a.new))}
    turns = {(t["tournament_id"], t["board"], t["key"]): t for t in map(json.loads, open(a.turns))}
    both = sorted(k for k in base if k in new and base[k].get("walk") and new[k].get("walk"))
    rows = []
    for k in both:
        wb, wn = base[k]["walk"], new[k]["walk"]
        d = dd[f"{k[0]}|{k[1]}"]
        cb, sb = result(wb, deals[k], d)
        cn, sn = result(wn, deals[k], d)
        rule, call, was = "", "", ""
        if wb != wn:
            i = next(j for j in range(min(len(wb), len(wn)) + 1) if j >= len(wb) or j >= len(wn) or wb[j] != wn[j])
            t = turns.get((k[0], k[1], "-".join(wn[:i])))
            call, was = wn[i] if i < len(wn) else "", wb[i] if i < len(wb) else ""
            rule = "other"
            if t:
                named = [p for r in a.rules.split(",") for p in t["props"] if p["rule"] == r and p["legal"] and p["bid"] == call]
                if named:
                    rule = named[0]["rule"] if not named[0]["rule"].startswith(("sacrifice", "double2")) else named[0]["rule"] + ": " + named[0]["tag"]
        rows.append({"k": k, "rule": rule, "call": call, "was": was, "base": (cb, sb), "new": (cn, sn), "diff": sn - sb,
                     "scoring": deals[k].get("scoring", "IMP"), "hand": deals[k]["deal"][2:].split()[2],
                     "auction": "-".join(wn), "auction_base": "-".join(wb)})
    diff = np.array([r["diff"] for r in rows], dtype=float)
    im = imps(diff)
    mp = 50 + 50 * np.sign(diff)
    out = []
    say = out.append
    changed = [r for r in rows if r["rule"]]
    say(f"{len(rows)} deals bid by both; auction differs on {len(changed)} ({len(changed) / len(rows):.1%}), "
        f"score on {int((diff != 0).sum())}")
    say(f"new against baseline, double dummy: {im.mean():+.3f} +-{ci(im):.3f} IMPs a deal, "
        f"{mp.mean():.2f}% +-{ci(mp):.2f} matchpoints; better on {int((diff > 0).sum())}, worse on {int((diff < 0).sum())}")
    for sc in ("MP", "IMP"):
        sel = np.array([r["scoring"] == sc for r in rows])
        if sel.any():
            say(f"  deals bid to {sc} scoring ({int(sel.sum())}): {im[sel].mean():+.3f} +-{ci(im[sel]):.3f} IMPs, "
                f"{mp[sel].mean():.2f}% +-{ci(mp[sel]):.2f}")
    say("")
    say(f"{'changed by':<22}{'deals':>6}{'better':>8}{'worse':>7}{'same':>6}{'IMPs sum':>10}{'IMPs each':>11}{'+-':>6}"
        f"{'IMPs a deal, all deals':>24}")
    groups = defaultdict(list)
    for r, x in zip(rows, im):
        if r["rule"]:
            groups[r["rule"]].append((r, x))
    for name in sorted(groups, key=lambda g: -len(groups[g])):
        g = groups[name]
        x = np.array([v for _, v in g])
        say(f"{name:<22}{len(g):>6}{int((x > 0).sum()):>8}{int((x < 0).sum()):>7}{int((x == 0).sum()):>6}{x.sum():>+10.0f}"
            f"{x.mean():>+11.2f}{ci(x):>6.2f}{x.sum() / len(rows):>+24.4f}")
    say("")
    say("Largest swings (hand, dealer, vul; baseline auction -> contract, score; new auction -> contract, score):")
    for name in sorted(groups, key=lambda g: -len(groups[g])):
        g = sorted(groups[name], key=lambda t: t[1])
        say(f" {name}")
        for r, x in (g if len(g) <= 2 * a.examples else g[:a.examples] + g[-a.examples:]):
            b = deals[r["k"]]
            say(f"   {x:+4.0f} IMPs  {r['hand']:<17} {b['dealer']} {b['vul']:<4} {r['auction_base']} -> {r['base'][0]} {r['base'][1]:+d}"
                f"  |  {r['auction']} -> {r['new'][0]} {r['new'][1]:+d}")
    text = "\n".join(out)
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
