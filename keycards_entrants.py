"""What each Brill entrant did at the key-card decision points of keycards.py, against choosing by the model's odds.

  bbo-bidding\\.venv\\Scripts\\python.exe keycards_entrants.py            (Windows, numpy only; seconds)

Every table of `src\\data\\brill-replay\\field.jsonl` is matched to its row (deal + auction) of `brill_test.npz` /
`brill_train.npz`; our model's replay `runs\\old.jsonl` is matched the same way where its auction is one an entrant
had (name `ours`). A row's decisions come from `keycards\\<tag>\\states.npz` and `decisions.jsonl`: the decision is at
South's last turn where the choice was still available, so the table's final contract shows what its South (and the
robot North after it) chose:
  partial or game, game or slam   higher = our side in the decision's strain at the higher level or above,
                                  lower = our side in that strain below it
  suit game or 3NT                higher = the suit at game or above, lower = notrump at 3 or above
  other                           anything else (another strain, the other side declares, passed out); left out of the IMPs
IMPs lost = IMPs short of the double-dummy right one of the two contracts on the real deal, the same measure as
keycards.py gives "choosing by the odds" and "choosing with the key set known" at that decision point.
Output: `E:\\bridge\\data\\auction-model\\keycards\\entrants.txt`.
"""
import collections
import json
import pathlib

import numpy as np

from extract_bbo import CALL_TOKEN, parse_pbn

DATA = pathlib.Path("E:/bridge/data/auction-model")
REPLAY = pathlib.Path(__file__).resolve().parent.parent / "data" / "brill-replay"
TAGS = {"test": "brill_test.npz", "train-v1": "brill_train.npz"}
VUL = {"None": 0, "NS": 1, "N_S": 1, "EW": 2, "E_W": 2, "All": 3, "Both": 3}
CALL = {"P": "p", "X": "d", "XX": "r", "PASS": "p"}
STRAINS = "SHDCN"
GAME = [4, 4, 5, 5, 3]
TYPES = ("PG", "GS", "SN")
TYPE_NAME = {"PG": "partial or game", "GS": "game or slam", "SN": "suit game or 3NT"}
OURS = "ours (GIB-BBO-8730-base)"
SHOW = ["BBOBen", "Thorvald", "DownTwo", "Brill", "BEN", OURS]


def wmean(x, g):
    """Mean and twice its standard error over deals."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan")
    m = x.mean()
    _, inv = np.unique(np.asarray(g), return_inverse=True)
    num = np.bincount(inv, x - m)
    G = len(num)
    return m, 2 * np.sqrt(G / max(G - 1, 1) * (num ** 2).sum()) / len(x)


def main():
    rows, dec, row_dec = {}, {}, {}
    for tag, f in TAGS.items():
        z = np.load(DATA / f)
        S = np.load(DATA / "keycards" / tag / "states.npz")
        for r in range(len(z["dealer"])):
            n = int(z["n_calls"][r])
            rows[(z["owner"][r].tobytes(), int(z["dealer"][r]), int(z["vul"][r]), z["calls"][r, :n].tobytes())] = (tag, r)
        row_dec[tag] = {k: S[f"row_{k}"] for k in TYPES}
        dec[tag] = {}
        for line in open(DATA / "keycards" / tag / "decisions.jsonl", encoding="utf-8"):
            x = json.loads(line)
            dec[tag][(x["type"], x["u"])] = x

    def decisions_of(owner, dealer, vul, auction):
        toks = [CALL_TOKEN.get(CALL.get(c.upper(), c.upper())) for c in auction]
        if None in toks:
            return None
        hit = rows.get((owner.tobytes(), dealer, vul, np.array(toks, np.uint8).tobytes()))
        if hit is None:
            return None
        tag, r = hit
        return [dec[tag][(k, int(row_dec[tag][k][r]))] for k in TYPES if row_dec[tag][k][r] >= 0]

    def final_contract(auction, dealer):
        """-> (level, DD strain, declarer side is NS) from the calls, or None when passed out."""
        last = None
        for i, c in enumerate(auction):
            c = c.upper()
            if c[0].isdigit():
                last = (int(c[0]), STRAINS.index(c[1]), (dealer + i) % 2 == 0)
        return last

    def choice(x, con):
        if con is None or not con[2]:
            return "other"
        level, strain, _ = con
        hs = STRAINS.index(x["high"][1])
        if x["type"] == "SN":
            if strain == hs and level >= GAME[hs]:
                return "high"
            return "low" if strain == 4 and level >= 3 else "other"
        if strain != hs:
            return "other"
        return "high" if level >= int(x["high"][0]) else "low"

    recs = collections.defaultdict(list)          # entrant -> (type, deal, choice, decision)
    tables = collections.Counter()
    unmatched = collections.Counter()
    boards = {}
    for line in open(REPLAY / "field.jsonl", encoding="utf-8"):
        b = json.loads(line)
        owner = parse_pbn(b["deal"])
        if owner is None or b["vul"] not in VUL:
            continue
        dealer = "NESW".index(b["dealer"])
        boards[(b["tournament_id"], b["board"])] = (owner, dealer, VUL[b["vul"]], b["deal"])
        for t in b["tables"]:
            if not t.get("auction"):
                continue
            ds = decisions_of(owner, dealer, VUL[b["vul"]], t["auction"])
            if ds is None:
                unmatched[t["name"]] += 1
                continue
            tables[t["name"]] += 1
            con = final_contract(t["auction"], dealer)
            for x in ds:
                recs[t["name"]].append((x["type"], b["deal"], choice(x, con), x))
    for line in open(REPLAY / "runs" / "old.jsonl", encoding="utf-8"):
        w = json.loads(line)
        bd = boards.get((w["tournament_id"], w["board"]))
        if bd is None:
            continue
        ds = decisions_of(bd[0], bd[1], bd[2], w["walk"])
        if ds is None:
            unmatched[OURS] += 1
            continue
        tables[OURS] += 1
        con = final_contract(w["walk"], bd[1])
        for x in ds:
            recs[OURS].append((x["type"], bd[3], choice(x, con), x))

    names = SHOW + [n for n, c in tables.most_common() if n not in SHOW and c >= 1000]
    out = ["keycards_entrants: each entrant's final contract at the decision points of keycards.py (double dummy, real deals).",
           "IMPs lost are short of the right one of the two contracts; 'odds' and 'set' are the model choosing by its odds and",
           "with the key set known, on the same tables. +- is twice the standard error over boards. Entrants with 1,000+ tables,",
           f"plus {', '.join(SHOW)}.", ""]
    total = {}
    for k in TYPES:
        out.append(f"=== {TYPE_NAME[k]}")
        out.append(f"  {'entrant':<26}{'tables':>7}{'decision':>9}{'other':>7}{'went higher':>19}{'right':>15}{'IMPs lost':>25}{'entrant - odds':>18}")
        out.append(f"  {'':<26}{'':>7}{'':>9}{'':>7}{'entrant / odds':>19}{'entrant / odds':>15}{'entrant / odds / set':>25}")
        for n in names:
            rr = [r for r in recs[n] if r[0] == k]
            use = [r for r in rr if r[2] != "other"]
            if not use:
                continue
            hi = np.array([r[2] == "high" for r in use])
            d = np.array([r[3]["d_true"] for r in use])
            now = np.array([r[3]["now"] for r in use])
            best = np.maximum(d, 0)
            l_e, l_o, l_k = best - d * hi, np.array([r[3]["loss_now"] for r in use]), np.array([r[3]["loss_key"] for r in use])
            nz = d != 0
            right = lambda c: ((c == (d > 0)) & nz).sum() / max(nz.sum(), 1)
            diff, se = wmean(l_e - l_o, [r[1] for r in use])
            total.setdefault(n, []).extend(zip(l_e - l_o, [r[1] for r in use], l_e - l_k))
            out.append(f"  {n:<26}{tables[n]:>7,}{len(rr) / tables[n]:>9.1%}{1 - len(use) / len(rr):>7.1%}"
                       f"{hi.mean():>10.1%} /{now.mean():>6.1%}{right(hi):>8.1%} /{right(now):>5.1%}"
                       f"{l_e.mean():>11.2f} /{l_o.mean():>5.2f} /{l_k.mean():>5.2f}{diff:>+10.2f} +-{se:.2f}")
        out.append("")
    out.append("=== All three decisions, IMPs per board over all the entrant's boards: what the entrant loses to the by-odds choice,")
    out.append("    and to the choice with the key set known (the most that knowing the key set could add to its bidding)")
    for n in names:
        if n not in total:
            continue
        x = np.array([v[0] for v in total[n]])
        y = np.array([v[2] for v in total[n]])
        g = [v[1] for v in total[n]]
        m, se = wmean(x, g)
        m2, se2 = wmean(y, g)
        f = len(x) / tables[n]
        out.append(f"  {n:<26}to odds {m * f:>+7.3f} +-{se * f:.3f}    to key set {m2 * f:>+7.3f} +-{se2 * f:.3f}   "
                   f"({len(x):,} decisions on {tables[n]:,} boards)")
    out.append("")
    out.append("=== Same board, same kind of decision, both went higher or lower: IMPs lost, first minus second")
    for a, b in ((OURS, "BBOBen"), ("BEN", "BBOBen"), (OURS, "BEN"), (OURS, "Brill"), ("Thorvald", "BBOBen")):
        parts = []
        for k in TYPES + ("all",):
            la = {(r[0], r[1]): max(r[3]["d_true"], 0) - r[3]["d_true"] * (r[2] == "high") for r in recs[a] if r[2] != "other" and k in (r[0], "all")}
            lb = {(r[0], r[1]): max(r[3]["d_true"], 0) - r[3]["d_true"] * (r[2] == "high") for r in recs[b] if r[2] != "other" and k in (r[0], "all")}
            common = sorted(set(la) & set(lb))
            m, se = wmean([la[c] - lb[c] for c in common], [c[1] for c in common])
            parts.append(f"{TYPE_NAME.get(k, 'all')}: {m:+.2f} +-{se:.2f} ({len(common):,})")
        out.append(f"  {a} - {b}:  " + ";  ".join(parts))
    out.append("")
    out.append("tables not matched to a row (auction not in the data files): " + ", ".join(f"{n} {unmatched[n]}" for n in names if unmatched[n]))
    txt = "\n".join(out)
    (DATA / "keycards" / "entrants.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
