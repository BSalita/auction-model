"""Key cards: at South's turns of the Brill auctions, which few cards or suit lengths of the hidden hands decide
between two contracts of our side, how likely they are to be there, and is that right on the real deals.

Runs in the `auction-model` container (PyTorch from /var/lib/ben/torch-pkgs, DDS from /opt/ben/bin/dds3-linux):

  export PYTHONPATH=/var/lib/ben/torch-pkgs
  python keycards.py --run /data/runs/v1-brill --data /data/brill_test.npz --tag test          # all three stages
  python keycards.py --tag test --stage report                                              # the report again

Stages, each saved under /data/keycards/<tag>/ and skipped when its file exists (--force redoes it):

  states  Every distinct South turn (deal + calls seen). The auction model's direct trick probabilities say which
          decisions are open there, for the best suit (highest expected tricks) and notrump:
            PG  partial or game    the game makes 20-80% of the time and a lower contract in the strain is available
            GS  game or slam       the small slam makes 20-80% and game (or the lowest level left) is available
            SN  suit game or 3NT   both games available, each makes 25%+, and one of them at most 85%
          A decision counts once per auction (row): at South's last turn where the choice is still available and
          partner has made a call other than pass, if it is open there. --n sets (suit lengths and A K Q J of the
          three hidden hands) are drawn from the model for each such turn.
  solve   Each set is completed to a deal (cards below the jack dealt at random within the lengths) and solved
          double dummy for our declarer in the suit and in notrump. About 41 ms a solve.
  report  The finder. For each decision d = IMPs gained by the higher contract (PG, GS) or the suit game (SN) over
          the other, on each deal. Features: a missing honor with partner / LHO / RHO, partner's length in a suit
          (3+ .. 6+, at most 1), partner's aces and kings (1+, 2+, 3+), partner's top-three trumps (1+, 2+).
          On the first third of the deals the set of 1 to 5 features is chosen whose presence, if known, is worth
          the most (IMPs gained by choosing the contract after learning it, over choosing by the odds). It is kept
          only if knowing it also changes the choice on the second third; the last third gives its probability and
          the results with and without it. Then the real deal: is the set there, which
          contract is right, what do "choose by the odds", "choose knowing the set" and "choose knowing the result"
          score. Everything is double dummy.

Memory: under 4 GB of GPU, 1 GB a solver process. Time: states 1-2 min, solve about 35 s of one core a turn kept.
"""
import argparse
import datetime
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from scoring import SCORE

DDS_PATH = "/opt/ben/bin/dds3-linux"
SUITS = "SHDC"
STRAINS = "SHDCN"
RANKS = "AKQJT98765432"
HONORS = [s + r for s in SUITS for r in "AKQJ"]
TOK_TO_DD = [3, 2, 1, 0, 4]            # call tokens use C,D,H,S,N; DD uses S,H,D,C,N
TOK_OF_DD = [3, 2, 1, 0, 4]
GAME = [4, 4, 5, 5, 3]
IMP_T = np.array([20, 50, 90, 130, 170, 220, 270, 320, 370, 430, 500, 600, 750, 900, 1100, 1300, 1500, 1750, 2000,
                  2250, 2500, 3000, 3500, 4000])
TYPES = ("PG", "GS", "SN")
TYPE_NAME = {"PG": "partial or game", "GS": "game or slam", "SN": "suit game or 3NT"}


def imps(x):
    return np.sign(x) * np.searchsorted(IMP_T, np.abs(x), side="right")


def call_str(t):
    if t < 4:
        return {1: "p", 2: "d", 3: "r"}[int(t)]
    return f"{(t - 4) // 5 + 1}{'CDHSN'[(t - 4) % 5]}"


def auction_info(calls, t, dealer):
    """-> lowest level our side can still play each DD strain at (8 = none), our declarer (absolute seat) per strain."""
    hl, htok, hby, first = 0, -1, -1, {}
    for i in range(t):
        c = int(calls[i])
        if c >= 4:
            hl, htok, hby = (c - 4) // 5 + 1, (c - 4) % 5, (dealer + i) % 4
            if hby % 2 == 0:
                first.setdefault(TOK_TO_DD[htok], hby)
    low = []
    for s in range(5):
        tok = TOK_OF_DD[s]
        low.append(1 if hl == 0 else hl if (hby % 2 == 0 and htok == tok) else hl if tok > htok else hl + 1)
    return low, [first.get(s, 2) for s in range(5)]


# ---------------------------------------------------------------------------------------------- stage: states

def stage_states(a, path):
    import torch
    from model import AuctionModel, sample_sets, view_inputs
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    ck = torch.load(os.path.join(a.run, "model.pt"), map_location=dev, weights_only=False)
    model = AuctionModel(**ck["cfg"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    z = np.load(a.data)
    owner, dealer, vul, calls, n_calls = z["owner"], z["dealer"].astype(int), z["vul"].astype(int), z["calls"], z["n_calls"].astype(int)
    R = len(dealer)
    _, deal_id = np.unique(owner, axis=0, return_inverse=True)
    deal_id = deal_id.reshape(-1)
    key2u, u_row, u_t, row_states = {}, [], [], [[] for _ in range(R)]
    for r in range(R):
        for t in range(n_calls[r]):
            if (dealer[r] + t) % 4 != 2:
                continue
            k = (int(deal_id[r]), dealer[r], vul[r], calls[r, :t].tobytes())
            if k not in key2u:
                key2u[k] = len(u_row)
                u_row.append(r)
                u_t.append(t)
            row_states[r].append(key2u[k])
    u_row, u_t = np.array(u_row), np.array(u_t)
    U = len(u_row)
    print(f"{R:,} rows on {deal_id.max() + 1:,} deals, {U:,} distinct South turns", flush=True)

    # encoder state of every distinct turn
    H, Fh, mylen, myhon = [None] * U, [None] * U, [None] * U, [None] * U
    by_row = {}
    for u in range(U):
        by_row.setdefault(int(u_row[u]), []).append(u)
    rows = sorted(by_row)
    with torch.no_grad():
        for i in range(0, len(rows), 512):
            rr = rows[i:i + 512]
            L = int(n_calls[rr].max())
            ct = torch.tensor(calls[rr, :L].astype(np.int64), device=dev)
            feats, callers, lens, hon = view_inputs(torch.tensor(owner[rr], device=dev), torch.tensor(dealer[rr], device=dev),
                                                    torch.tensor(vul[rr], device=dev), ct)
            pick = torch.arange(len(rr), device=dev) * 4 + 2
            hh, ff = model.encode(feats[pick], ct, callers[pick])
            for j, r in enumerate(rr):
                for u in by_row[r]:
                    H[u], Fh[u] = hh[j, int(u_t[u])], ff[j]
                    mylen[u], myhon[u] = lens[pick[j], :, 0], hon[pick[j]] == 0
        H, Fh, mylen, myhon = torch.stack(H), torch.stack(Fh), torch.stack(mylen), torch.stack(myhon)
        dd = torch.cat([model.dd_head(H[i:i + 4096]).float().view(-1, 4, 5, 14).softmax(-1) for i in range(0, U, 4096)]).cpu().numpy()

    # which decisions are open at each turn
    low, decl = np.zeros((U, 5), np.int8), np.zeros((U, 5), np.int8)
    s_star = np.zeros(U, np.int8)
    strain = {k: np.full(U, -1, np.int8) for k in ("PG", "GS")}
    live = {k: np.zeros(U, bool) for k in TYPES}
    avail = {k: np.zeros(U, bool) for k in TYPES}
    p_now = {k: np.zeros(U, np.float32) for k in TYPES}
    ar = np.arange(14)
    for u in range(U):
        r = int(u_row[u])
        lo, de = auction_info(calls[r], int(u_t[u]), dealer[r])
        low[u], decl[u] = lo, de
        # partner (North) has made a call other than pass: before that the choice is not South's to make
        acted = any(calls[r, i] != 1 for i in range((0 - dealer[r]) % 4, int(u_t[u]), 4))
        v = int(bool(vul[r] & 1))
        p = np.stack([dd[u, (de[s] - 2) % 4, s] for s in range(5)])                       # (5,14) our tricks by strain
        ss = int((p[:4] * ar).sum(1).argmax())
        s_star[u] = ss
        make = lambda lvl, s: float(p[s, lvl + 6:].sum())
        ev = lambda lvl, s: float((p[s] * SCORE[lvl - 1, s, 0, v]).sum())
        c = [s for s in (ss, 4) if lo[s] < GAME[s]]
        if c:
            x = max(c, key=lambda s: ev(GAME[s], s))
            strain["PG"][u], p_now["PG"][u] = x, make(GAME[x], x)
            avail["PG"][u] = acted
            live["PG"][u] = acted and 0.2 <= p_now["PG"][u] <= 0.8
        c = [s for s in (ss, 4) if max(lo[s], GAME[s]) < 6]
        if c:
            x = max(c, key=lambda s: ev(6, s))
            strain["GS"][u], p_now["GS"][u] = x, make(6, x)
            avail["GS"][u] = acted
            live["GS"][u] = acted and 0.2 <= p_now["GS"][u] <= 0.8
        if lo[ss] <= GAME[ss] and lo[4] <= 3:
            pa, pb = make(GAME[ss], ss), make(3, 4)
            p_now["SN"][u] = pa
            avail["SN"][u] = acted
            live["SN"][u] = acted and min(pa, pb) >= 0.25 and min(pa, pb) <= 0.85

    # per row and type: South's last turn where the choice is still available; it counts when it is open there
    row_last = {k: np.full(R, -1, np.int64) for k in TYPES}
    for r in range(R):
        for k in TYPES:
            last = -1
            for u in row_states[r]:
                if avail[k][u]:
                    last = u
            if last >= 0 and live[k][last]:
                row_last[k][r] = last
    keep = np.unique(np.concatenate([row_last[k][row_last[k] >= 0] for k in TYPES]))
    print("turns with the decision open: " + ", ".join(f"{k} {int(live[k].sum()):,}" for k in TYPES)
          + f"; kept (South's last turn with the choice, open there): {len(keep):,}", flush=True)

    N = a.n
    lens_s = np.zeros((len(keep), N, 4, 3), np.int8)
    hon_s = np.zeros((len(keep), N, 16), np.int8)
    step = max(1, 8192 // N)
    t0 = time.time()
    for i in range(0, len(keep), step):
        idx = torch.tensor(keep[i:i + step], device=dev)
        rep = lambda x: x[idx].repeat_interleave(N, 0)
        _, sl, sh, _, _ = sample_sets(model, rep(H), rep(Fh), rep(mylen), rep(myhon))
        lens_s[i:i + step] = sl.view(-1, N, 4, 3).cpu().numpy()
        hon_s[i:i + step] = sh.view(-1, N, 16).cpu().numpy()
    print(f"{len(keep) * N:,} sets drawn in {time.time() - t0:.0f}s", flush=True)
    np.savez(path, u_row=u_row, u_t=u_t, deal_id=deal_id, low=low, decl=decl, s_star=s_star, keep=keep,
             lens=lens_s, hon=hon_s, **{f"strain_{k}": v for k, v in strain.items()},
             **{f"live_{k}": v for k, v in live.items()}, **{f"p_{k}": v for k, v in p_now.items()},
             **{f"row_{k}": v for k, v in row_last.items()})


# ---------------------------------------------------------------------------------------------- stage: solve

W = {}


def solve_init():
    sys.path.insert(0, DDS_PATH)
    import dds3
    W["dds"], W["ctx"] = dds3, dds3.SolverContext()


def solve_state(a):
    i, mine, lens, hon, cells, seed = a
    dds, ctx = W["dds"], W["ctx"]
    rng = np.random.default_rng(seed)
    n = lens.shape[0]
    out = np.zeros((n, len(cells)), np.uint8)
    my = [[r for r in range(13) if mine[s * 13 + r]] for s in range(4)]
    loose = [np.array([r for r in range(4, 13) if r not in my[s]], np.int64) for s in range(4)]
    for k in range(n):
        hands = [[[] for _ in range(4)] for _ in range(4)]                 # absolute seat -> suit -> ranks
        for s in range(4):
            hands[2][s] = list(my[s])
            cnt = [0, 0, 0]
            for j in range(4):
                r = int(hon[k, s * 4 + j])
                if r > 0:
                    hands[(2 + r) % 4][s].append(j)
                    cnt[r - 1] += 1
            cards = rng.permutation(loose[s])
            p = 0
            for r in (1, 2, 3):
                room = int(lens[k, s, r - 1]) - cnt[r - 1]
                hands[(2 + r) % 4][s] += [int(x) for x in cards[p:p + room]]
                p += room
            assert p == len(loose[s]), "set does not fit the cards"
        pbn = "N:" + " ".join(".".join("".join(RANKS[x] for x in sorted(h[s])) for s in range(4)) for h in hands)
        for c, (de, st) in enumerate(cells):
            r = dds.solve_board_pbn(pbn, trump=int(st), first=(int(de) + 1) % 4, current_trick_suit=(0, 0, 0),
                                    current_trick_rank=(0, 0, 0), target=-1, solutions=1, mode=1, context=ctx)
            out[k, c] = 13 - r["score"][0]
    return i, out


def stage_solve(a, S, path):
    z = np.load(a.data)
    keep = S["keep"]
    tasks = []
    for i, u in enumerate(keep):
        r = int(S["u_row"][u])
        ss = int(S["s_star"][u])
        cells = [(int(S["decl"][u, ss]), ss), (int(S["decl"][u, 4]), 4)]
        tasks.append((i, z["owner"][r] == 2, S["lens"][i], S["hon"][i], cells, 5000 + int(u)))
    tricks = np.zeros((len(keep), S["lens"].shape[1], 2), np.uint8)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, solve_init) as pool:
        for j, (i, out) in enumerate(pool.imap_unordered(solve_state, tasks, chunksize=1)):
            tricks[i] = out
            if (j + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {j + 1:,} of {len(tasks):,} turns solved, {el:.0f}s, about {el / (j + 1) * (len(tasks) - j - 1):.0f}s left", flush=True)
    np.savez(path, tricks=tricks)


# ---------------------------------------------------------------------------------------------- stage: report

def build_features(lens, hon, myhon, trump):
    """lens (n,4,3) for LHO, partner, RHO; hon (n,16) seat of each honor (0 me, 1 LHO, 2 partner, 3 RHO)."""
    cols, names = [], []
    for j in range(16):
        if myhon[j]:
            continue
        for r, who in ((2, "partner"), (1, "LHO"), (3, "RHO")):
            cols.append(hon[:, j] == r)
            names.append((who, HONORS[j], "honor"))
    for s in range(4):
        x = lens[:, s, 1]
        for k in (3, 4, 5, 6):
            cols.append(x >= k)
            names.append(("partner", f"{k}+ {SUITS[s]}", "length"))
        cols.append(x <= 1)
        names.append(("partner", f"at most 1 {SUITS[s]}", "shortness"))
    p = hon == 2
    aces, kings = p[:, [0, 4, 8, 12]].sum(1), p[:, [1, 5, 9, 13]].sum(1)
    for k in (1, 2, 3):
        cols.append(aces >= k)
        names.append(("partner", f"{k}+ aces", "count"))
    for k in (1, 2):
        cols.append(kings >= k)
        names.append(("partner", f"{k}+ kings", "count"))
    if trump < 4:
        top = p[:, trump * 4:trump * 4 + 3].sum(1)
        for k in (1, 2):
            cols.append(top >= k)
            names.append(("partner", f"{k}+ of {SUITS[trump]} A K Q", "count"))
    return np.stack(cols, 1), names


def find_set(X, d, min_n=10, top=16, max_size=5):
    """Feature set whose presence, if known, gains the most over choosing by the odds. X (n,F) bool, d (n,) IMPs."""
    n = len(d)
    n1 = X.sum(0)
    ok = (n1 >= min_n) & (n - n1 >= min_n)
    if not ok.any():
        return ()
    m1 = (X * d[:, None]).sum(0) / np.maximum(n1, 1)
    m0 = (d.sum() - m1 * n1) / np.maximum(n - n1, 1)
    ss = np.where(ok, n1 * (n - n1) / n * (m1 - m0) ** 2, -1.0)
    cand = [int(c) for c in np.argsort(-ss)[:top] if ok[c]]
    base = max(d.mean(), 0.0)

    def value(combo):
        m = X[:, list(combo)].all(1)
        k = int(m.sum())
        if k < min_n or n - k < min_n:
            return -1.0
        return (max(d[m].sum(), 0.0) + max(d[~m].sum(), 0.0)) / n - base - 0.05 * (len(combo) - 1)

    best, best_v = (), 0.1
    for size in (1, 2, 3):
        for combo in itertools.combinations(cand, size):
            v = value(combo)
            if v > best_v:
                best, best_v = combo, v
    while best and len(best) < max_size:
        ext = max(((value(best + (c,)), c) for c in cand if c not in best), default=(-1.0, None))
        if ext[0] <= best_v + 0.1:
            break
        best, best_v = best + (ext[1],), ext[0]
    return best


def wmean(x, w, g):
    """Weighted mean and twice its standard error over deals."""
    if w.sum() == 0:
        return float("nan"), float("nan")
    m = (w * x).sum() / w.sum()
    _, inv = np.unique(g, return_inverse=True)
    num = np.bincount(inv, w * (x - m))
    G = len(num)
    return m, 2 * np.sqrt(G / max(G - 1, 1) * (num ** 2).sum()) / w.sum()


def stage_report(a, S, T, out_dir):
    z = np.load(a.data)
    owner, dealer, vul, calls, n_calls, dd, count = (z["owner"].astype(int), z["dealer"].astype(int), z["vul"].astype(int),
                                                      z["calls"], z["n_calls"].astype(int), z["dd"].astype(int), z["count"].astype(float))
    R = len(dealer)
    keep = S["keep"]
    pos = {int(u): i for i, u in enumerate(keep)}
    N = S["lens"].shape[1]
    third = N // 3
    dec = {}                                                   # (type, u) -> record
    for k in TYPES:
        for u in np.unique(S[f"row_{k}"][S[f"row_{k}"] >= 0]):
            u = int(u)
            i, r = pos[u], int(S["u_row"][u])
            v = int(bool(vul[r] & 1))
            ss = int(S["s_star"][u])
            lo, de = S["low"][u].astype(int), S["decl"][u].astype(int)
            tr = T["tricks"][i].astype(int)                                         # (N, 2): suit cell, notrump cell
            true_t = [dd[r, de[ss] * 5 + ss], dd[r, de[4] * 5 + 4]]
            if k == "SN":
                hi, lw = (GAME[ss], ss, 0), (3, 4, 1)
                trump = ss
            else:
                x = int(S[f"strain_{k}"][u])
                cell = 0 if x == ss else 1
                hi = (GAME[x] if k == "PG" else 6, x, cell)
                lw = (lo[x] if k == "PG" else max(lo[x], GAME[x]), x, cell)
                trump = x
            sc = lambda con, t: SCORE[con[0] - 1, con[1], 0, v][t]
            d = imps(sc(hi, tr[:, hi[2]]) - sc(lw, tr[:, lw[2]])).astype(float)
            d_true = float(imps(sc(hi, true_t[hi[2]]) - sc(lw, true_t[lw[2]])))
            if k == "SN":
                y, y_true = d > 0, d_true > 0
            else:
                y, y_true = tr[:, hi[2]] >= hi[0] + 6, true_t[hi[2]] >= hi[0] + 6
            myhon = owner[r, [s * 13 + j for s in range(4) for j in range(4)]] == 2
            X, names = build_features(S["lens"][i].astype(int), S["hon"][i].astype(int), myhon, trump)
            combo = find_set(X[:third], d[:third])
            if combo:                                   # second third: knowing the set must change the choice there too
                m = X[third:2 * third, list(combo)].all(1)
                e = d[third:2 * third]
                if m.sum() < 5 or (~m).sum() < 5 or (e[m].mean() > 0) == (e[~m].mean() > 0):
                    combo = ()
            lens_t = np.stack([[(owner[r, s * 13:(s + 1) * 13] == (2 + rel) % 4).sum() for rel in (1, 2, 3)] for s in range(4)])[None]
            hon_t = ((owner[r, [s * 13 + j for s in range(4) for j in range(4)]] - 2) % 4)[None]
            Xt, _ = build_features(lens_t, hon_t, myhon, trump)
            rec = {"type": k, "u": u, "row": r, "deal": int(S["deal_id"][r]), "t": int(S["u_t"][u]),
                   "high": f"{hi[0]}{STRAINS[hi[1]]}", "low": f"{lw[0]}{STRAINS[lw[1]]}",
                   "p_direct": float(S[f"p_{k}"][u]), "p_high": float(y.mean()), "ev_now": float(d.mean()),
                   "d_true": d_true, "y_true": bool(y_true), "set": [f"{names[c][0]} {names[c][1]}" for c in combo],
                   "kinds": [names[c][2] if names[c][2] != "honor" else ("partner honor" if names[c][0] == "partner" else "opponent honor") for c in combo]}
            now = d.mean() > 0
            rec["now"] = bool(now)
            if combo:
                m = X[2 * third:, list(combo)].all(1)
                e = d[2 * third:]
                if m.sum() < 5 or (~m).sum() < 5:                       # too few in the last third: use the last two
                    m, e, yy = X[third:, list(combo)].all(1), d[third:], y[third:]
                else:
                    yy = y[2 * third:]
                rec.update(p_set=float(m.mean()), mk1=float(yy[m].mean()), mk0=float(yy[~m].mean()),
                           d1=float(e[m].mean()), d0=float(e[~m].mean()), set_true=bool(Xt[0, list(combo)].all()))
                key = (rec["d1"] > 0) if rec["set_true"] else (rec["d0"] > 0)
                rec["voi"] = float(m.mean() * max(e[m].mean(), 0) + (1 - m.mean()) * max(e[~m].mean(), 0) - max(e.mean(), 0))
            else:
                key = now
            rec["key"] = bool(key)
            best = max(d_true, 0.0)
            rec["loss_now"], rec["loss_key"] = best - d_true * now, best - d_true * key
            rec["hand"] = ".".join("".join(RANKS[j] for j in range(13) if owner[r, s * 13 + j] == 2) for s in range(4))
            rec["auction"] = " ".join(call_str(c) for c in calls[r, :rec["t"]]) or "(none)"
            rec["dealer"], rec["vul"] = "NESW"[dealer[r]], ("None", "NS", "EW", "Both")[vul[r]]
            dec[(k, u)] = rec

    deal = S["deal_id"]
    n_deals = int(deal.max()) + 1
    out = [f"keycards {datetime.datetime.now().isoformat(timespec='seconds')}: {os.path.basename(a.data)}, model {a.run}; "
           f"{R:,} auctions (rows) at {int(count.sum()):,} tables on {n_deals:,} boards; {N} deals a decision, double dummy",
           "A decision is counted once per auction: at South's last turn where the choice is still available and partner has",
           "bid or doubled, if it is open there. Numbers are weighted by tables; +- is twice the standard error over boards."]
    out.append("\n=== How many boards hold such a decision, and what is at stake (IMPs, on the real deal)")
    out.append(f"  {'decision':<18}{'boards':>8}{'tables':>8}{'higher makes':>22}{'apart':>7}{'lost by odds':>14}{'lost with set':>15}"
               f"{'set gains':>16}{'per board, all':>16}")
    out.append(f"  {'':<18}{'':>8}{'':>8}{'said -> true':>22}")
    tot_gain = np.zeros(R)
    tot_ceiling = np.zeros(R)
    any_live = np.zeros(R, bool)
    per_type = {}
    for k in TYPES:
        ru = S[f"row_{k}"]
        lv = ru >= 0
        g = lambda f: np.array([dec[(k, int(u))][f] if u >= 0 else 0.0 for u in ru], float)
        loss_now, loss_key, d_true, p_high, y_true = g("loss_now"), g("loss_key"), g("d_true"), g("p_high"), g("y_true")
        per_type[k] = (lv, ru)
        any_live |= lv
        tot_gain += (loss_now - loss_key) * lv
        tot_ceiling += loss_now * lv
        w = count * lv
        gain, se = wmean(loss_now - loss_key, w, deal)
        ga, sea = wmean((loss_now - loss_key) * lv, count, deal)
        out.append(f"  {TYPE_NAME[k]:<18}{len(np.unique(deal[lv])) / n_deals:>8.1%}{w.sum() / count.sum():>8.1%}"
                   f"{wmean(p_high, w, deal)[0]:>12.1%} ->{wmean(y_true, w, deal)[0]:>7.1%}{wmean(np.abs(d_true), w, deal)[0]:>7.2f}"
                   f"{wmean(loss_now, w, deal)[0]:>14.2f}{wmean(loss_key, w, deal)[0]:>15.2f}{gain:>+9.2f} +-{se:.2f}{ga:>+9.3f} +-{sea:.3f}")
    ga, sea = wmean(tot_gain, count, deal)
    gc, sec = wmean(tot_ceiling, count, deal)
    out.append(f"  {'any of the three':<18}{len(np.unique(deal[any_live])) / n_deals:>8.1%}{(count * any_live).sum() / count.sum():>8.1%}"
               f"{'':>58}{'':>16}{ga:>+9.3f} +-{sea:.3f}")
    out.append(f"  ceiling (choosing with the double-dummy result known, over choosing by the odds), per board over all boards: {gc:+.3f} +-{sec:.3f}")
    out.append("  apart: IMPs between the two contracts. lost by odds / with set: IMPs short of the right contract when choosing by the")
    out.append("  model's odds / after learning whether the key set is there. set gains: the difference, per decision and per board.")

    out.append("\n=== Is the finder right (decisions where a key set was found)")
    out.append(f"  {'decision':<18}{'set found':>10}{'items':>7}{'set is there':>20}{'makes with it':>20}{'makes without':>20}{'worth, said':>13}")
    out.append(f"  {'':<18}{'':>10}{'':>7}{'said -> true':>20}{'said -> true':>20}{'said -> true':>20}")
    for k in TYPES:
        lv, ru = per_type[k]
        recs = [dec[(k, int(u))] if u >= 0 else None for u in ru]
        has = np.array([bool(x and x["set"]) for x in recs])
        g = lambda f: np.array([x[f] if (x and x["set"]) else 0.0 for x in recs], float)
        w = count * has
        st = g("set_true") > 0
        y = np.array([bool(x["y_true"]) if x else False for x in recs], float)
        out.append(f"  {TYPE_NAME[k]:<18}{w.sum() / max((count * lv).sum(), 1):>10.1%}"
                   f"{wmean(np.array([len(x['set']) if x else 0 for x in recs], float), w, deal)[0]:>7.2f}"
                   f"{wmean(g('p_set'), w, deal)[0]:>10.1%} ->{wmean(st.astype(float), w, deal)[0]:>7.1%}"
                   f"{wmean(g('mk1'), w * st, deal)[0]:>10.1%} ->{wmean(y, w * st, deal)[0]:>7.1%}"
                   f"{wmean(g('mk0'), w * ~st, deal)[0]:>10.1%} ->{wmean(y, w * ~st, deal)[0]:>7.1%}{wmean(g('voi'), w, deal)[0]:>13.2f}")
    out.append("  makes: the higher contract makes (partial or game, game or slam); the suit game scores more than 3NT (suit game or 3NT).")

    kinds = {}
    for (k, u), x in dec.items():
        for kd in x["kinds"]:
            kinds[kd] = kinds.get(kd, 0) + 1
    tot = max(sum(kinds.values()), 1)
    out.append("\n=== What the key sets are made of: " + ", ".join(f"{k} {v / tot:.0%}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
    sizes = np.bincount([len(x["set"]) for x in dec.values()], minlength=6)
    out.append("  items in a set (0 = none found): " + ", ".join(f"{i}: {c / len(dec):.0%}" for i, c in enumerate(sizes)))

    out.append("\n=== Examples (largest said worth first, one per board)")
    seen, shown = set(), 0
    for x in sorted((x for x in dec.values() if x["set"]), key=lambda x: -x["voi"]):
        if x["deal"] in seen:
            continue
        seen.add(x["deal"])
        out.append(f"  South {x['hand']}, dealer {x['dealer']}, vul {x['vul']}, auction {x['auction']}: {x['low']} or {x['high']}")
        out.append(f"    key set: {'; '.join(x['set'])}  (there {x['p_set']:.0%}; {x['high'] if x['type'] != 'SN' else 'suit game better'} "
                   f"{x['mk1']:.0%} with it, {x['mk0']:.0%} without; {x['d1']:+.1f} / {x['d0']:+.1f} IMPs)")
        out.append(f"    real deal: set {'there' if x['set_true'] else 'not there'}, {x['high']} gains {x['d_true']:+.0f} IMPs over {x['low']}")
        shown += 1
        if shown == 12:
            break
    txt = "\n".join(out)
    open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8").write(txt + "\n")
    with open(os.path.join(out_dir, "decisions.jsonl"), "w", encoding="utf-8") as f:
        for x in dec.values():
            f.write(json.dumps(x) + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/data/runs/v1-brill")
    ap.add_argument("--data", default="/data/brill_test.npz")
    ap.add_argument("--tag", default="test")
    ap.add_argument("--stage", default="all", choices=["all", "states", "solve", "report"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out_dir = os.path.join("/data/keycards", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    sp, tp = os.path.join(out_dir, "states.npz"), os.path.join(out_dir, "tricks.npz")
    t0 = time.time()
    print("start", datetime.datetime.now().isoformat(timespec="seconds"), flush=True)
    if a.stage in ("all", "states") and (a.force or not os.path.exists(sp)):
        stage_states(a, sp)
    if a.stage in ("all", "solve") and (a.force or not os.path.exists(tp)):
        stage_solve(a, np.load(sp), tp)
    if a.stage in ("all", "report"):
        stage_report(a, np.load(sp), np.load(tp), out_dir)
    print(f"end {datetime.datetime.now().isoformat(timespec='seconds')}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
