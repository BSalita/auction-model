"""The BBO bidding table (bbo-bidding, 461M auction nodes with hand requirements per seat) as a source of deals
for South's turns of the Brill test auctions, against BEN's own sampled deals and against the true deal.

  D:\\...\\src\\bbo-bidding\\.venv\\Scripts\\python.exe bt_check.py            (Windows; polars and numpy, no PyTorch)

For every state of `ben_samples_old.npz` (row of `brill_test.npz`, calls seen, South to call):
  1. the auction so far, leading passes stripped, is looked up in `bbo_bt_seat1.parquet`; the table's seat 1 is the
     opener, so seats are rotated by the number of leading passes (as the bbo-bidding API does);
  2. each seat's requirements (`Agg_Expr_Seat_N`) are cleaned as the live API cleans them (a metric whose numeric
     bounds contradict each other is dropped) and evaluated: HCP, Total_Points (HCP + 3/2/1 for void, singleton,
     doubleton), QT, suit lengths, card held (C_HA), biddable / rebiddable / stopper names with the definitions of
     mlBridgeAugmentLib.HandAugmenter. Names that are not a property of a hand (Forcing_One_Round, ...) are ignored;
  3. do the true hidden hands meet the table's requirements;
  4. deals: the 39 unseen cards are dealt at random and kept when all three hidden hands meet their requirements
     (up to --tries deals tried, the first --k kept). The first --k random deals are kept too, as the
     "knows nothing" baseline.
The kept deals are scored exactly as `sample_check.py` scores BEN's (honor seats, points, suit lengths), on the
states where both BEN and the table gave deals. The custom overlay of bbo-bidding (`bbo_custom_auction_criteria.csv`)
is not applied.

Time: about 2 minutes with 20 processes. Memory: under 1 GB a process. Output: `bt_check.txt`, `bt_samples.npz`.
"""
import argparse
import collections
import datetime
import math
import multiprocessing as mp
import pathlib
import re
import time

import numpy as np

import paths

BT = pathlib.Path(paths.BBO) / "bidding" / "bbo_bt_seat1.parquet"
DATA = pathlib.Path(paths.DATA)
SUITS = "SHDC"
RANKS = "AKQJT98765432"
REL = ("LHO", "partner", "RHO")

# ---------------------------------------------------------------------------------------------- requirements

TOK = re.compile(r"\s*(\d+(?:\.\d+)?|[A-Za-z_]\w*|>=|<=|==|!=|>|<|[&|()+\-~!])")
CMP = {">=": np.greater_equal, "<=": np.less_equal, "==": np.equal, "!=": np.not_equal, ">": np.greater, "<": np.less}
INEQ = re.compile(r"^(\w+)\s*(>=|<=|>|<|==)\s*(-?\d+)$")


class NotAHandProperty(Exception):
    pass


def parse(text):
    """Requirement text -> tree. Precedence: not, comparison, &, |."""
    text = text.split("#")[0].strip()
    toks = TOK.findall(text)
    if "".join(toks) != re.sub(r"\s+", "", text) or not toks:
        raise ValueError(text)
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        pos[0] += 1
        return toks[pos[0] - 1]

    def p_or():
        n = p_and()
        while peek() is not None and peek().lower() in ("|", "or"):
            take()
            n = ("|", n, p_and())
        return n

    def p_and():
        n = p_not()
        while peek() is not None and peek().lower() in ("&", "and"):
            take()
            n = ("&", n, p_not())
        return n

    def p_not():
        if peek() is not None and peek().lower() in ("not", "~", "!"):
            take()
            return ("~", p_not())
        return p_cmp()

    def p_cmp():
        a = p_sum()
        if peek() in CMP:
            op = take()
            return (op, a, p_sum())
        return a

    def p_sum():
        a = p_atom()
        while peek() in ("+", "-"):
            op = take()
            a = (op, a, p_atom())
        return a

    def p_atom():
        t = take()
        if t == "(":
            n = p_or()
            if take() != ")":
                raise ValueError(text)
            return n
        if t[0].isdigit():
            return ("num", float(t))
        if t in ("True", "False"):
            return ("num", t == "True")
        return ("var", t)

    tree = p_or()
    if pos[0] != len(toks):
        raise ValueError(text)
    return tree


def ev(node, F):
    k = node[0]
    if k == "num":
        return node[1]
    if k == "var":
        if node[1] not in F:
            raise NotAHandProperty(node[1])
        return F[node[1]]
    if k == "~":
        return ~np.asarray(ev(node[1], F), bool)
    if k == "|":
        return np.asarray(ev(node[1], F), bool) | np.asarray(ev(node[2], F), bool)
    if k == "&":
        return np.asarray(ev(node[1], F), bool) & np.asarray(ev(node[2], F), bool)
    if k == "+":
        return ev(node[1], F) + ev(node[2], F)
    if k == "-":
        return ev(node[1], F) - ev(node[2], F)
    return CMP[k](ev(node[1], F), ev(node[2], F))


_trees = {}


def tree_of(text):
    if text not in _trees:
        try:
            _trees[text] = parse(text)
        except Exception:
            _trees[text] = None
    return _trees[text]


def features(H):
    """H (B,52) bool, card = suit*13 + rank -> every name a requirement can use."""
    X = H.reshape(-1, 4, 13)
    sl = X.sum(-1)
    hs = (X[:, :, :4] * np.array([4, 3, 2, 1])).sum(-1)
    hcp = hs.sum(-1)
    dp = ((sl == 0) * 3 + (sl == 1) * 2 + (sl == 2) * 1).sum(-1)
    A, K, Q = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    qt = np.where(A & K, 2.0, np.where(A & Q, 1.5, np.where(A, 1.0, np.where(K & Q, 1.0, np.where(K & (sl >= 2), 0.5, 0.0))))).sum(-1)
    F = {"HCP": hcp, "Total_Points": hcp + dp, "QT": qt}
    for i, s in enumerate(SUITS):
        l, h = sl[:, i], hs[:, i]
        F["SL_" + s] = l
        F["HCP_" + s] = h
        F["Biddable_" + s] = (l >= 5) | ((l == 4) & (h >= 3))
        F["Rebiddable_" + s] = (l >= 6) | ((l == 5) & (h >= 3))
        F["Twice_Rebiddable_" + s] = (l >= 7) | ((l == 6) & (h >= 3))
        F["Strong_Rebiddable_" + s] = (l >= 6) & (h >= 9)
        F["Solid_" + s] = h >= 9
        F["At_Best_Partial_Stop_In_" + s] = (l + h) < 4
        F["Partial_Stop_In_" + s] = (l + h) >= 4
        F["Likely_Stop_In_" + s] = (l + h) >= 5
        F["Stop_In_" + s] = (h >= 4) | ((l + h) >= 6)
        F["At_Best_Stop_In_" + s] = (l + h) >= 7
        F["Two_Stops_In_" + s] = (l + h) >= 8
        for j, r in enumerate(RANKS):
            F[f"C_{s}{r}"] = X[:, i, j]
    return F


def clean(criteria):
    """As bbo-bidding's _dedupe_criteria_least_restrictive_with_notes: least restrictive bound per metric and
    operator; a metric whose bounds leave no value is dropped. Returns (kept list, a metric was dropped)."""
    ineq, eqs, other = {}, collections.defaultdict(set), []
    for c in criteria or []:
        c = str(c).strip()
        m = INEQ.match(c)
        if not m:
            if c and c not in other:
                other.append(c)
            continue
        var, op, v = m.group(1), m.group(2), int(m.group(3))
        if op == "==":
            eqs[var].add(v)
        key = (var, op)
        if key not in ineq:
            ineq[key] = v
        elif op in (">=", ">"):
            ineq[key] = min(ineq[key], v)
        elif op in ("<=", "<"):
            ineq[key] = max(ineq[key], v)
    bad = set()
    for var in {k[0] for k in ineq}:
        lo, hi = -math.inf, math.inf
        if (var, ">=") in ineq:
            lo = max(lo, ineq[(var, ">=")])
        if (var, ">") in ineq:
            lo = max(lo, ineq[(var, ">")] + 1)
        if (var, "<=") in ineq:
            hi = min(hi, ineq[(var, "<=")])
        if (var, "<") in ineq:
            hi = min(hi, ineq[(var, "<")] - 1)
        if len(eqs[var]) > 1:
            bad.add(var)
            continue
        if eqs[var]:
            e = next(iter(eqs[var]))
            lo, hi = max(lo, e), min(hi, e)
        if lo > hi:
            bad.add(var)
    return sorted(f"{var} {op} {v}" for (var, op), v in ineq.items() if var not in bad) + other, bool(bad)


def check(criteria, F):
    """-> (all met (B,) bool, list of (text, met array) for the requirements that are hand properties, ignored names)."""
    ok, each, ignored = None, [], []
    for c in criteria:
        t = tree_of(c)
        if t is None:
            ignored.append(c)
            continue
        try:
            r = np.asarray(ev(t, F), bool)
        except NotAHandProperty as e:
            ignored.append(str(e))
            continue
        each.append((c, r))
        ok = r if ok is None else ok & r
    return ok, each, ignored


# ---------------------------------------------------------------------------------------------- one state

def call_tok(t):
    if t < 4:
        return {1: "p", 2: "d", 3: "r"}[int(t)]
    return f"{(t - 4) // 5 + 1}{'CDHSN'[(t - 4) % 5]}"


def prefix_of(calls, t):
    toks = [call_tok(c) for c in calls[:t]]
    k = 0
    while k < len(toks) and toks[k] == "p":
        k += 1
    return "-".join(toks[k:]), k


def do_state(a):
    i, owner, dealer, k_pass, node, no_bid, K, tries, batch = a
    rng = np.random.default_rng(1000 + i)
    unseen = np.flatnonzero(owner != 2)
    pos = np.full(52, -1)
    pos[unseen] = np.arange(39)
    honors = np.array([s * 13 + r for s in range(4) for r in range(4)])
    out = {"i": i, "found": node is not None, "n": 0, "tried": 0, "accepted": 0, "dropped": [False] * 4,
           "true_ok": [True] * 3, "south_ok": True, "fails": [], "applied": [], "ignored": []}

    def table_seat(direction):
        return ((direction - dealer) % 4 - k_pass) % 4 + 1

    crit = [[], [], []]
    if node is not None:
        for r in range(3):
            crit[r], out["dropped"][r] = clean(node[table_seat((3 + r) % 4) - 1])
        south, out["dropped"][3] = clean(node[table_seat(2) - 1])
        # the true hands against the table
        for r in range(3):
            H = (owner == (3 + r) % 4)[None, :]
            ok, each, ign = check(crit[r], features(H))
            out["ignored"] += ign
            out["true_ok"][r] = True if ok is None else bool(ok[0])
            for c, m in each:
                key = (REL[r], re.sub(r"\d+", "#", c))
                out["applied"].append(key)
                if not m[0]:
                    out["fails"].append(key)
        ok, _, _ = check(south, features((owner == 2)[None, :]))
        out["south_ok"] = True if ok is None else bool(ok[0])

    def deal(B):
        soc = rng.permuted(np.tile(np.repeat(np.arange(3), 13), (B, 1)), axis=1)      # seat (0 LHO, 1 partner, 2 RHO) of each unseen card
        return soc

    def pack(soc):
        n = soc.shape[0]
        H = np.zeros((n, 3, 52), bool)
        for r in range(3):
            H[:, r, unseen] = soc == r
        lens = H.reshape(n, 3, 4, 13).sum(-1).transpose(0, 2, 1).astype(np.int8)       # (n, suit, rel seat)
        hon = np.zeros((n, 16), np.int8)
        for j, c in enumerate(honors):
            if pos[c] >= 0:
                hon[:, j] = soc[:, pos[c]] + 1
        return lens, hon

    first = deal(K)
    out["rand"] = pack(first)
    kept = []
    if node is not None:
        got = 0
        while out["tried"] < tries and got < K:
            soc = deal(batch)
            ok = np.ones(batch, bool)
            for r in range(3):
                if not crit[r]:
                    continue
                H = np.zeros((batch, 52), bool)
                H[:, unseen] = soc == r
                m, _, _ = check(crit[r], features(H))
                if m is not None:
                    ok &= m
                if not ok.any():
                    break
            out["tried"] += batch
            out["accepted"] += int(ok.sum())
            if ok.any():
                kept.append(soc[ok][: K - got])
                got += kept[-1].shape[0]
    lens = np.zeros((K, 4, 3), np.int8)
    hon = np.zeros((K, 16), np.int8)
    if no_bid:                                   # nothing to look up: the table says nothing, its deals are random deals
        out["n"] = K
        lens, hon = out["rand"]
    if kept:
        soc = np.concatenate(kept)[:K]
        out["n"] = soc.shape[0]
        lens[: out["n"]], hon[: out["n"]] = pack(soc)
    out["bt"] = (lens, hon)
    return out


# ---------------------------------------------------------------------------------------------- scoring

HCP16 = np.array([4.0, 3.0, 2.0, 1.0] * 4)


def source_stats(hon_s, lens_s, n, hon, lens):
    """As sample_check.source_stats. hon_s (R,S,16) seat 0..3, lens_s (R,S,4,3), n (R,) deals; truth hon (R,16), lens (R,4,3)."""
    R, S = hon_s.shape[:2]
    valid = np.arange(S)[None, :] < n[:, None]
    w = valid / np.maximum(n, 1)[:, None]
    hon_p = np.stack([((hon_s == v) * w[:, :, None]).sum(1) for v in range(4)], -1)                 # (R,16,4)
    len_p = np.stack([((lens_s == v) * w[:, :, None, None]).sum(1) for v in range(14)], -1)         # (R,4,3,14)
    hidden = hon > 0
    y = np.stack([hon == v for v in range(4)], -1).astype(float)
    m = hidden.astype(float)
    n_h = np.maximum(m.sum(1), 1)
    o = {}
    o["honor seat error (Brier)"] = (((hon_p[:, :, 1:] - y[:, :, 1:]) ** 2).sum(-1) * m).sum(1) / n_h / 3
    o["most likely seat of an honor right"] = ((hon_p[:, :, 1:].argmax(-1) + 1 == hon) * m).sum(1) / n_h
    o["honor given under 10% is there"] = (hon_p[:, :, 1:], y[:, :, 1:], hidden)
    e_hcp = (hon_p[:, :, 1:] * (HCP16 * m)[:, :, None]).sum(1)
    t_hcp = (y[:, :, 1:] * (HCP16 * m)[:, :, None]).sum(1)
    o["points of a hidden hand off by"] = np.abs(e_hcp - t_hcp).mean(1)
    e_len = (len_p * np.arange(14)).sum(-1)
    o["suit length off by (cards)"] = np.abs(e_len - lens).mean((1, 2))
    p_true = np.take_along_axis(len_p, lens[..., None].astype(np.int64), -1)[..., 0]
    o["probability of the true length"] = p_true.mean((1, 2))
    o["true length in none of the deals"] = (p_true == 0).mean((1, 2))
    return o


def mean_pm(x, row_id):
    _, inv = np.unique(row_id, return_inverse=True)
    s, c = np.bincount(inv, x), np.bincount(inv)
    g = len(s)
    return x.mean(), 2 * np.sqrt(g / max(g - 1, 1) * ((s - x.mean() * c) ** 2).sum()) / c.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA / "brill_test.npz"))
    ap.add_argument("--ben", default=str(DATA / "ben_samples_old.npz"))
    ap.add_argument("--out", default=str(DATA / "bt_check.txt"))
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--tries", type=int, default=400_000)
    ap.add_argument("--batch", type=int, default=20_000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    print("start", datetime.datetime.now().isoformat(timespec="seconds"), flush=True)
    import polars as pl

    D = np.load(a.data)
    B = np.load(a.ben)
    rows, ts = B["row"], B["t"]
    if a.limit:
        rows, ts = rows[: a.limit], ts[: a.limit]
    N = len(rows)
    pref = [prefix_of(D["calls"][r], int(t)) for r, t in zip(rows, ts)]
    wanted = sorted({p for p, _ in pref if p})
    cols = ["Auction"] + [f"Agg_Expr_Seat_{s}" for s in range(1, 5)]
    bt = pl.scan_parquet(BT).filter(pl.col("Auction").is_in(wanted)).select(cols).collect()
    nodes = {r[0]: [list(x or []) for x in r[1:]] for r in bt.iter_rows()}
    print(f"{N:,} states, {len(wanted):,} distinct auctions looked up, {len(nodes):,} found, {time.time() - t0:.0f}s", flush=True)

    args = [(i, D["owner"][rows[i]], int(D["dealer"][rows[i]]), pref[i][1], nodes.get(pref[i][0]), pref[i][0] == "", a.k, a.tries, a.batch)
            for i in range(N)]
    res = [None] * N
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for j, o in enumerate(pool.imap_unordered(do_state, args, chunksize=8)):
            res[o["i"]] = o
            if (j + 1) % 500 == 0:
                print(f"  {j + 1:,} of {N:,} states, {time.time() - t0:.0f}s", flush=True)

    dealer = D["dealer"][rows].astype(int)
    turn = (ts.astype(int) - (2 - dealer) % 4) // 4 + 1
    no_bid = np.array([p == "" for p, _ in pref])
    found = np.array([o["found"] for o in res])
    n_bt = np.array([o["n"] for o in res])
    tried = np.array([o["tried"] for o in res])
    acc = np.array([o["accepted"] for o in res])
    true_ok = np.array([o["true_ok"] for o in res])
    south_ok = np.array([o["south_ok"] for o in res])
    dropped = np.array([o["dropped"] for o in res])
    owner = D["owner"][rows].astype(int)
    hon_t = (owner[:, [s * 13 + r for s in range(4) for r in range(4)]] - 2) % 4
    lens_t = np.stack([np.stack([(owner[:, s * 13:(s + 1) * 13] == (2 + rel) % 4).sum(1) for rel in (1, 2, 3)], -1) for s in range(4)], 1)
    bt_lens = np.stack([o["bt"][0] for o in res])
    bt_hon = np.stack([o["bt"][1] for o in res])
    rd_lens = np.stack([o["rand"][0] for o in res])
    rd_hon = np.stack([o["rand"][1] for o in res])
    np.savez(DATA / "bt_samples.npz", row=rows, t=ts, n=n_bt.astype(np.int16), lens=bt_lens, hon=bt_hon,
             tried=tried, accepted=acc, found=found, true_ok=true_ok)

    out = [f"bt_check {datetime.datetime.now().isoformat(timespec='seconds')}: {N:,} South turns on {len(np.unique(rows)):,} rows of "
           f"{pathlib.Path(a.data).name}; table {BT.name}; {a.k} deals a turn, up to {a.tries:,} random deals tried"]
    groups = [("All South turns", np.ones(N, bool)), ("South's first turn", turn == 1), ("South's second turn", turn == 2),
              ("South's third or later turn", turn >= 3)]

    out.append("\n=== The table against the true hands")
    out.append(f"  {'':<30}{'turns':>7}{'no bid yet':>11}{'in table':>10}{'LHO fits':>10}{'partner':>9}{'RHO fits':>10}"
               f"{'all three':>11}{'South fits':>12}{'kept of tried':>15}{'50 deals':>10}{'1-49':>7}{'none':>7}")
    for title, g in groups:
        b = g & ~no_bid
        f = b & found
        pct = lambda x, d: f"{x.sum() / max(d.sum(), 1):.1%}"
        out.append(f"  {title:<30}{g.sum():>7,}{no_bid[g].sum():>11,}{pct(f, b):>10}{pct(f & true_ok[:, 0], f):>10}{pct(f & true_ok[:, 1], f):>9}"
                   f"{pct(f & true_ok[:, 2], f):>10}{pct(f & true_ok.all(1), f):>11}{pct(f & south_ok, f):>12}"
                   f"{acc[f].sum() / max(tried[f].sum(), 1):>15.2%}{pct(f & (n_bt >= a.k), f):>10}{pct(f & (n_bt > 0) & (n_bt < a.k), f):>7}{pct(f & (n_bt == 0), f):>7}")
    f = found & ~no_bid
    out.append(f"  a seat's requirements held a contradiction that was dropped: LHO {dropped[f, 0].mean():.1%}, partner {dropped[f, 1].mean():.1%}, "
               f"RHO {dropped[f, 2].mean():.1%}, South {dropped[f, 3].mean():.1%} of turns in the table")
    out.append(f"  median share of random deals that meet the requirements: {np.median(acc[f] / np.maximum(tried[f], 1)):.2%}")

    ap_c, fl_c = collections.Counter(), collections.Counter()
    ig = collections.Counter()
    for o in res:
        ap_c.update(o["applied"])
        fl_c.update(o["fails"])
        ig.update(re.sub(r"\d+", "#", x) for x in o["ignored"])
    out.append("\n=== Requirements the true hand does not meet (seat, requirement with numbers as #: failed of applied)")
    for key, v in sorted(fl_c.items(), key=lambda kv: -kv[1])[:24]:
        out.append(f"  {key[0]:<8} {key[1]:<34} {v:>6,} of {ap_c[key]:>6,}  {v / ap_c[key]:.1%}")
    out.append("  not a property of a hand, ignored: " + ", ".join(f"{k} ({v:,})" for k, v in ig.most_common(12)))

    both = (B["n"][:N] > 0) & (n_bt > 0)
    S = {"random deals": source_stats(rd_hon, rd_lens, np.full(N, a.k), hon_t, lens_t),
         "table deals": source_stats(bt_hon, bt_lens, n_bt, hon_t, lens_t),
         "BEN's deals": source_stats(B["hon"][:N], B["lens"][:N], B["n"][:N].astype(int), hon_t, lens_t)}
    names = list(S)
    out.append(f"\n=== Deals against the true deal, on the {both.sum():,} turns where BEN and the table both gave deals "
               f"(BEN none on {(B['n'][:N] == 0).sum()}, table none on {(n_bt == 0).sum()})")
    for title, g in groups + [("Turns where all three true hands fit the table", true_ok.all(1) & ~no_bid),
                              ("Turns where a true hand does not fit", ~true_ok.all(1))]:
        mk = g & both
        if not mk.any():
            continue
        out.append(f"\n--- {title}: {mk.sum():,} turns")
        out.append(f"  {'measure':<38}" + "".join(f"{n:>15}" for n in names) + f"{'table - BEN':>24}")
        for key in S[names[0]]:
            v = [S[n][key] for n in names]
            if isinstance(v[0], tuple):
                cells = []
                for p, y, hid in v:
                    sel = (p < 0.1) & hid[:, :, None] & mk[:, None, None]
                    cells.append(f"{y[sel].mean():>14.1%} ")
                out.append(f"  {key:<38}" + "".join(cells))
                continue
            pct = "right" in key or "probability" in key or "none" in key
            fmt = (lambda x: f"{x:>14.1%} ") if pct else (lambda x: f"{x:>14.4f} " if "Brier" in key else f"{x:>14.3f} ")
            d, se = mean_pm((v[1] - v[2])[mk], rows[mk])
            dtxt = f"{d:+.1%} +-{se:.1%}" if pct else f"{d:+.4f} +-{se:.4f}"
            out.append(f"  {key:<38}" + "".join(fmt(x[mk].mean()) for x in v) + f"{dtxt:>24}")
    out.append(f"\nelapsed {time.time() - t0:.0f}s")
    txt = "\n".join(out)
    pathlib.Path(a.out).write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print("end", datetime.datetime.now().isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()
