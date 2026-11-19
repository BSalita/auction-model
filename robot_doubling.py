"""How often do the Brill robots double a contract of ours that fails, and how often are failing contracts of theirs doubled?
Final contracts of the live replays (South = BEN config, N/E/W = Brill's service), double dummy."""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from scoring import STRAINS, contract_score
from tactical import SEATS, Auction
VUL = {"None": [False, False], "NS": [True, False], "EW": [False, True], "All": [True, True]}
L = os.path.join(paths.BRILL, "live")
deals = {(b["tournament_id"], b["board"]): b for b in map(json.loads, open(L + r"\deals10k.jsonl"))}
dd = json.load(open(L + r"\10k-report\dd.json"))
seen, rows = set(), []
for name in sys.argv[1:]:
    for line in open(rf"{L}\10k\{name}\replay.jsonl"):
        r = json.loads(line)
        k = (r["tournament_id"], r["board"])
        key = k + ("-".join(r["walk"]),)
        if key in seen:
            continue
        seen.add(key)
        b = deals[k]
        dealer = SEATS.index(b["dealer"])
        a = Auction(r["walk"], dealer, (dealer + len(r["walk"])) % 4)
        if a.last is None:
            continue
        _, level, strain, who = a.last
        decl = a.declarer(who, strain)
        d = dd[f"{k[0]}|{k[1]}"]
        tricks = d["tricks"][strain + SEATS[decl]]
        vul = VUL[b["vul"]][decl % 2]
        sc = contract_score(level, STRAINS.index(strain), a.dbl, vul, tricks)
        und = contract_score(level, STRAINS.index(strain), 0, vul, tricks)
        ns = decl % 2 == 0
        rows.append(dict(ns=ns, level=level, down=max(0, level + 6 - tricks), dbl=a.dbl > 0, vul=vul,
                         vs_par=(sc if ns else -sc) - d["par"], gain_dbl=abs(sc) - abs(und) if tricks < level + 6 else und - sc,
                         contested=len({w % 2 for _, _, _, w in a.bids}) == 2, scoring=b.get("scoring", "?"),
                         hcp=sum(4 * h.count("A") + 3 * h.count("K") + 2 * h.count("Q") + h.count("J") for h in (b["deal"][2:].split()[0], b["deal"][2:].split()[2]))))
print(len(rows), "distinct final contracts from", sys.argv[1:])
def table(sel, title):
    print(title)
    print(f"  {'':<22}{'n':>6}{'doubled':>9}{'vs par, doubled':>17}{'vs par, not':>13}")
    def line(label, g):
        if not g: return
        x = [r for r in g if r["dbl"]]; y = [r for r in g if not r["dbl"]]
        m = lambda z: f"{sum(r['vs_par'] for r in z) / len(z):+.0f}" if z else "-"
        print(f"  {label:<22}{len(g):>6}{100 * len(x) / len(g):>8.1f}%{m(x):>17}{m(y):>13}")
    for lab, f in (("makes", lambda r: r["down"] == 0), ("down 1", lambda r: r["down"] == 1), ("down 2", lambda r: r["down"] == 2),
                   ("down 3", lambda r: r["down"] == 3), ("down 4 or more", lambda r: r["down"] >= 4)):
        line(lab, [r for r in sel if f(r)])
    fail = [r for r in sel if r["down"] > 0]
    for lab, f in (("fails, level 1-2", lambda r: r["level"] <= 2), ("fails, level 3", lambda r: r["level"] == 3), ("fails, level 4", lambda r: r["level"] == 4),
                   ("fails, level 5+", lambda r: r["level"] >= 5), ("fails, we vul" if sel and sel[0]["ns"] else "fails, they vul", lambda r: r["vul"]),
                   ("fails, not vul", lambda r: not r["vul"]), ("fails, both sides bid", lambda r: r["contested"]), ("fails, one side bid", lambda r: not r["contested"])):
        line(lab, [r for r in fail if f(r)])
    line("fails, down 2+, both bid", [r for r in fail if r["down"] >= 2 and r["contested"]])
    line("fails, down 3+, both bid", [r for r in fail if r["down"] >= 3 and r["contested"]])

print("North-South declare, both sides bid: share doubled by down count and North-South points")
print(f"  {'':<10}" + "".join(f"{c:>16}" for c in ("NS up to 17", "NS 18-22", "NS 23+", "all")))
for lab, f in (("makes", lambda r: r["down"] == 0), ("down 1", lambda r: r["down"] == 1), ("down 2", lambda r: r["down"] == 2), ("down 3", lambda r: r["down"] == 3), ("down 4+", lambda r: r["down"] >= 4)):
    line = f"  {lab:<10}"
    for g in (lambda r: r["hcp"] <= 17, lambda r: 18 <= r["hcp"] <= 22, lambda r: r["hcp"] >= 23, lambda r: True):
        sel = [r for r in rows if r["ns"] and r["contested"] and f(r) and g(r)]
        line += f"{(f'{100 * sum(r[chr(100) + chr(98) + chr(108)] for r in sel) / len(sel):.0f}% of {len(sel)}' if sel else '-'):>16}"
    print(line)

print("Share of failing final contracts that end doubled, by the board's scoring")
print(f"  {'':<44}{'MP':>16}{'IMP':>16}")
for lab, f in (("ours, doubled by the robots: down 1", lambda r: r["ns"] and r["down"] == 1), ("ours: down 2", lambda r: r["ns"] and r["down"] == 2),
               ("ours: down 3 or more", lambda r: r["ns"] and r["down"] >= 3), ("ours: any, both sides bid", lambda r: r["ns"] and r["down"] > 0 and r["contested"]),
               ("theirs, doubled by North-South: down 1", lambda r: not r["ns"] and r["down"] == 1), ("theirs: down 2", lambda r: not r["ns"] and r["down"] == 2),
               ("theirs: down 3 or more", lambda r: not r["ns"] and r["down"] >= 3),
               ("ours, makes (doubled wrongly)", lambda r: r["ns"] and r["down"] == 0), ("theirs, makes (doubled wrongly)", lambda r: not r["ns"] and r["down"] == 0)):
    line = f"  {lab:<44}"
    for sc in ("MP", "IMP"):
        g = [r for r in rows if f(r) and r["scoring"] == sc]
        line += f"{(f'{100 * sum(r[chr(100) + chr(98) + chr(108)] for r in g) / len(g):.1f}% of {len(g)}' if g else '-'):>16}"
    print(line)
