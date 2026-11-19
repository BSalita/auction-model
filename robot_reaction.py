"""What the Brill robots do after North-South bid over a contract of theirs: double, bid on, or pass it out.
By what they held (game or partscore), how many tricks our bid is down double dummy, and our side's points.

  python robot_reaction.py live old-brillall-noadj old mix2-pure tac tac-even      (live replays, today's engine)
  python robot_reaction.py field                                                   (past Brill events, every entrant)
  python robot_reaction.py table old-brillall-noadj old mix2-pure tac tac-even     (both pooled -> robot_reaction.json,
                                                                                    the rates tactical.py prices with)
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
from tactical import GAME, SEATS, Auction  # noqa: E402

BR = paths.BRILL


def points(deal):
    h = deal[2:].split()
    return sum(4 * x.count("A") + 3 * x.count("K") + 2 * x.count("Q") + x.count("J") for x in (h[0], h[2]))


def auctions(mode, names):
    if mode == "live":
        deals = {(b["tournament_id"], b["board"]): b for b in map(json.loads, open(BR + r"\live\deals10k.jsonl"))}
        dd = json.load(open(BR + r"\live\10k-report\dd.json"))
        for name in names:
            for line in open(rf"{BR}\live\10k\{name}\replay.jsonl"):
                r = json.loads(line)
                k = (r["tournament_id"], r["board"])
                yield k, deals[k], dd[f"{k[0]}|{k[1]}"], r["walk"]
    else:
        dd = json.load(open(os.path.join(paths.DATA, "tactical", "dd_brill.json")))
        for line in open(BR + r"\field.jsonl"):
            b = json.loads(line)
            d = dd.get(f"{b['tournament_id']}|{b['board']}")
            if d:
                for t in b["tables"]:
                    yield (b["tournament_id"], b["board"]), b, d, t["auction"]


def collect(mode, names):
    seen, rows = set(), []
    for k, b, d, walk in auctions(mode, names):
        key = k + ("-".join(walk),)
        if key in seen:
            continue
        seen.add(key)
        dealer = SEATS.index(b["dealer"])
        a = Auction(walk, dealer, (dealer + len(walk)) % 4)
        ns = [x for x in a.bids if x[3] % 2 == 0]
        if not ns:
            continue
        i, level, strain, who = ns[-1]
        before = [x for x in a.bids if x[0] < i]
        if not before or before[-1][3] % 2 == 0:
            continue                                            # our last bid was not over a contract of theirs
        _, t_level, t_strain, _ = before[-1]
        after = a.calls[i + 1:]
        outcome = "bid on" if any(c[0].isdigit() for c in after) else "doubled" if "X" in after else "passed"
        decl = a.declarer(who, strain)
        down = max(0, level + 6 - d["tricks"][strain + SEATS[decl]])
        rows.append({"game": t_level >= GAME[t_strain], "down": min(down, 4), "outcome": outcome, "pts": points(b["deal"]),
                     "level": level, "scoring": b.get("scoring", "?")})
    return rows


def write_table(rows, path, k=25):
    """{"game" | "part": {"17" | "22" | "40": [[doubled, bid on] for 0..4+ tricks down]}}: the share of our bids the robots
    doubled / bid on over, by our side's points (up to 17, 18-22, 23+). A cell is pulled toward the rate of its
    row (all points) with the weight of k auctions, so thin cells do not jump."""
    out = {}
    for name, game in (("game", True), ("part", False)):
        out[name] = {}
        for top, lo in ((17, 0), (22, 18), (40, 23)):
            cells = []
            for down in range(5):
                marg = [r for r in rows if r["game"] == game and r["down"] == down]
                cell = [r for r in marg if lo <= r["pts"] <= top]
                rate = lambda g, o: sum(r["outcome"] == o for r in g) / len(g) if g else 0.0
                cells.append([round((len(cell) * rate(cell, o) + k * rate(marg, o)) / (len(cell) + k), 3) for o in ("doubled", "bid on")])
            out[name][str(top)] = cells
    json.dump(out, open(path, "w"), indent=1)
    for name in out:
        for top, cells in out[name].items():
            print(f"  {name:<5} our side up to {top:>2}: " + "  ".join(f"down {d}: X {c[0]:.0%} on {c[1]:.0%}" for d, c in enumerate(cells)))


def main():
    mode, names = sys.argv[1], sys.argv[2:]
    if mode == "table":
        rows = collect("live", names) + collect("field", [])
        print(f"{len(rows)} auctions pooled (live {' '.join(names)} + past events)")
        write_table(rows, os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_reaction.json"))
        return
    rows = collect(mode, names)
    print(f"{len(rows)} auctions where North-South's last bid was over a contract of East-West ({mode} {' '.join(names)})")

    def table(sel, title):
        print(title)
        print(f"  {'our bid, double dummy':<24}{'n':>6}{'doubled':>9}{'they bid on':>13}{'passed out':>12}{'doubled, of those left in':>27}")
        groups = [("makes", lambda r: r["down"] == 0), ("down 1", lambda r: r["down"] == 1), ("down 2", lambda r: r["down"] == 2),
                  ("down 3", lambda r: r["down"] == 3), ("down 4 or more", lambda r: r["down"] >= 4), ("all", lambda r: True),
                  ("fails, our side up to 17", lambda r: r["down"] > 0 and r["pts"] <= 17),
                  ("fails, our side 18-22", lambda r: r["down"] > 0 and 18 <= r["pts"] <= 22),
                  ("fails, our side 23+", lambda r: r["down"] > 0 and r["pts"] >= 23)]
        for lab, f in groups:
            g = [r for r in sel if f(r)]
            if len(g) < 10:
                continue
            c = defaultdict(int)
            for r in g:
                c[r["outcome"]] += 1
            left = c["doubled"] + c["passed"]
            print(f"  {lab:<24}{len(g):>6}{100 * c['doubled'] / len(g):>8.0f}%{100 * c['bid on'] / len(g):>12.0f}%{100 * c['passed'] / len(g):>11.0f}%"
                  f"{(f'{100 * c[chr(100) + chr(111) + chr(117) + chr(98) + chr(108) + chr(101) + chr(100)] / left:.0f}%' if left else '-'):>27}")

    print("By the board's scoring (failing bids only):")
    print(f"  {'':<34}{'n':>6}{'doubled':>9}{'they bid on':>13}{'passed out':>12}")
    for lab, g0 in (("over their game", [r for r in rows if r["game"]]), ("over their partscore", [r for r in rows if not r["game"]])):
        for sc in ("MP", "IMP"):
            for lab2, f in (("", lambda r: True), (", down 2 or more", lambda r: r["down"] >= 2)):
                g = [r for r in g0 if r["scoring"] == sc and r["down"] > 0 and f(r)]
                if len(g) >= 10:
                    c = lambda o: 100 * sum(r["outcome"] == o for r in g) / len(g)
                    print(f"  {lab + ', ' + sc + lab2:<34}{len(g):>6}{c('doubled'):>8.0f}%{c('bid on'):>12.0f}%{c('passed'):>11.0f}%")
    table([r for r in rows if r["game"]], "Over their game or higher (sacrifice or push):")
    table([r for r in rows if not r["game"]], "Over their partscore (competing):")


if __name__ == "__main__":
    main()
