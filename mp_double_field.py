"""The matchpoint double: Brill MP events, real field. For every table where East-West declare an undoubled contract,
matchpoint North-South's score against the other tables of the board for four cases: contract just makes / one down,
undoubled / doubled. Shows where doubling has little to lose and what it would have gained with the tricks really taken."""
import json, sys
import numpy as np
sys.path.insert(0, r"D:\OneDrive\sw\bridge\ML-Contract-Bridge\src\auction-model")
from scoring import STRAINS, contract_score
from tactical import SEATS, GAME, Auction
VUL = {"None": [False, False], "NS": [True, False], "EW": [False, True], "All": [True, True]}

def mp(s, field):
    f = np.asarray(field)
    return float(((f < s).sum() + 0.5 * (f == s).sum()) / len(f))

rows = []
for line in open(r"D:\OneDrive\sw\bridge\ML-Contract-Bridge\src\data\brill-replay\field.jsonl"):
    b = json.loads(line)
    if b["scoring"] != "MP" or len(b["tables"]) < 8:
        continue
    scores = [t["ns_score"] for t in b["tables"]]
    dealer = SEATS.index(b["dealer"])
    for i, t in enumerate(b["tables"]):
        k = t["contract"].upper().replace("NT", "N")
        if not k[:1].isdigit() or t["declarer"] not in ("E", "W") or "X" in k[2:]:
            continue
        level, strain = int(k[0]), k[1]
        vul = VUL[b["vul"]][1]
        field = scores[:i] + scores[i + 1:]
        sc = lambda dbl, tricks: -contract_score(level, STRAINS.index(strain), dbl, vul, tricks)
        a = Auction(t["auction"], dealer, (dealer + len(t["auction"])) % 4)
        ns = [x for x in a.bids if x[3] % 2 == 0]
        ew = [x for x in a.bids if x[3] % 2 == 1]
        pushed = bool(ns) and ew[-1][0] > ns[-1][0]
        before = [x for x in ew if x[0] < ns[-1][0]] if ns else []
        game = level >= GAME[strain]
        pushed_game = pushed and game and not any(x[1] >= GAME[x[2]] for x in before)
        def same(o):
            ko = o["contract"].upper().replace("NT", "N")
            return ko[:1].isdigit() and o["declarer"] in ("E", "W") and ko[1] == strain and int(ko[0]) >= level
        others = b["tables"][:i] + b["tables"][i + 1:]
        field_in = sum(same(o) for o in others) / len(others)
        tr = t["tricks"]
        rows.append(dict(pushed=pushed, game=game, pushed_game=pushed_game, vul=vul, level=level, failed=tr < level + 6, field_in=field_in,
                         make_u=mp(sc(0, level + 6), field), make_x=mp(sc(1, level + 6), field),
                         down_u=mp(sc(0, level + 5), field), down_x=mp(sc(1, level + 5), field),
                         real_u=mp(sc(0, tr), field), real_x=mp(sc(1, tr), field)))
R = rows
print(f"{len(R)} tables with East-West in an undoubled contract, on MP boards with 8+ tables; matchpoints are North-South's, 0-100.")
print("made / failed = with the tricks really taken at that table; X = the same tricks, doubled.")
print(f"{'class':<44}{'tables':>7}{'failed':>8}{'made':>7}{'made X':>8}{'failed':>8}{'failed X':>10}{'loss':>6}{'gain':>6}{'break-even':>12}{'always X':>12}")
def line(label, sel):
    made = [r for r in sel if not r["failed"]]
    fail = [r for r in sel if r["failed"]]
    if len(made) < 15 or len(fail) < 15:
        print(f"{label:<44}{len(sel):>7}   (too few)"); return
    m = lambda g, k: 100 * np.mean([r[k] for r in g])
    loss, gain = m(made, "real_u") - m(made, "real_x"), m(fail, "real_x") - m(fail, "real_u")
    d = [r["real_x"] - r["real_u"] for r in sel]
    print(f"{label:<44}{len(sel):>7}{100 * len(fail) / len(sel):>7.0f}%{m(made, 'real_u'):>7.1f}{m(made, 'real_x'):>8.1f}{m(fail, 'real_u'):>8.1f}"
          f"{m(fail, 'real_x'):>10.1f}{loss:>6.1f}{gain:>6.1f}{100 * loss / (loss + gain):>11.0f}%{100 * np.mean(d):>+7.1f}+-{196 * np.std(d, ddof=1) / np.sqrt(len(d)):.1f}")
line("all", R)
line("game or higher", [r for r in R if r["game"]])
line("partscore", [r for r in R if not r["game"]])
line("they bid on over our bid", [r for r in R if r["pushed"]])
line("pushed into game", [r for r in R if r["pushed_game"]])
line("they are vulnerable", [r for r in R if r["vul"]])
line("not vulnerable", [r for r in R if not r["vul"]])
print("by the share of the other tables where East-West are in the same strain at this level or higher:")
for lo, hi in ((0, 0.10), (0.10, 0.30), (0.30, 0.60), (0.60, 0.90), (0.90, 1.01)):
    line(f"  field in it {100 * lo:.0f}-{min(100, 100 * hi):.0f}%", [r for r in R if lo <= r["field_in"] < hi])
print("field in it under 30%:")
F = [r for r in R if r["field_in"] < 0.30]
line("  game or higher", [r for r in F if r["game"]])
line("  partscore", [r for r in F if not r["game"]])
line("  they bid on over our bid", [r for r in F if r["pushed"]])
line("  they are vulnerable", [r for r in F if r["vul"]])
line("  not vulnerable", [r for r in F if not r["vul"]])
for lab, sel in (("all", R), ("they bid on over our bid", [r for r in R if r["pushed"]]), ("pushed into game", [r for r in R if r["pushed_game"]])):
    f = np.array([r["field_in"] for r in sel])
    print(f"{lab}: {len(sel)} tables; field in it under 10%: {100 * (f < 0.10).mean():.0f}%, under 30%: {100 * (f < 0.30).mean():.0f}%, under 60%: {100 * (f < 0.60).mean():.0f}%, mean {100 * f.mean():.0f}%")
