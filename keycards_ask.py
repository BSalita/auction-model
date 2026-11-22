"""Is an asking bid worth it, on the real deals? At the game-or-slam decisions of keycards.py (South's turns of the
Brill auctions where the model's slam odds are 20-80%, a suit agreed, the auction still below 4NT), compare:

  odds    choose game or slam now by the model's odds (as keycards.py)
  rkcb    bid 4NT (RKCB 1430), read partner's reply from the real hand, then choose 5 or 6 of the suit by the odds
          of the deals drawn for that reply; 4 of the suit is gone
  key     keycards.py's key set, known for certain (its best case)
  dd      the right contract of the two on the real deal (ceiling)

and how well the model's reply odds match the real replies. Reuses keycards.py's saved sets and solved tricks
(states.npz, tricks.npz); seconds, numpy only.

  python keycards_ask.py --data /data/brill_test.npz --tag test
  python keycards_ask.py --data /data/brill_train.npz --tag train-v1
"""
import argparse
import json
import os

import numpy as np

from keycards import GAME, SCORE, STRAINS, imps, wmean

REPLIES = ["5C 1 or 4", "5D 0 or 3", "5H 2 no queen", "5S 2 and queen"]


def reply_of(hon, mine_q, t):
    """hon (n,16) relative seat of each honor (2 = partner) -> RKCB 1430 reply index for trump t."""
    kc = (hon[:, [0, 4, 8, 12]] == 2).sum(1) + (hon[:, t * 4 + 1] == 2)
    q = (hon[:, t * 4 + 2] == 2) | mine_q
    return np.where((kc == 1) | (kc >= 4), 0, np.where((kc == 0) | (kc == 3), 1, np.where(q, 3, 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/brill_test.npz")
    ap.add_argument("--tag", default="test")
    a = ap.parse_args()
    d_dir = os.path.join("/data/keycards", a.tag)
    S, T = np.load(os.path.join(d_dir, "states.npz")), np.load(os.path.join(d_dir, "tricks.npz"))
    z = np.load(a.data)
    owner, vul, calls, dd, count = z["owner"].astype(int), z["vul"].astype(int), z["calls"], z["dd"].astype(int), z["count"].astype(float)
    recs = {}
    for line in open(os.path.join(d_dir, "decisions.jsonl"), encoding="utf-8"):
        x = json.loads(line)
        if x["type"] == "GS":
            recs[x["u"]] = x
    pos = {int(u): i for i, u in enumerate(S["keep"])}
    rows = []
    said = np.zeros(4)
    true_hist = np.zeros(4)
    for u, x in recs.items():
        i, r = pos[u], int(S["u_row"][u])
        t_now = int(S["u_t"][u])
        st = int(S["strain_GS"][u])
        if st == 4:
            continue                                         # notrump: no keycard ask in this test
        high = max([int(c) for c in calls[r, :t_now] if c >= 4], default=0)
        if high >= 4 + 3 * 5 + 4:                            # 4NT or higher already bid: no room to ask
            continue
        lo = max(int(S["low"][u, st]), GAME[st])
        if lo >= 5:
            continue
        cell = 0 if st == int(S["s_star"][u]) else 1
        de = int(S["decl"][u, st])
        v = int(bool(vul[r] & 1))
        tr = T["tricks"][i][:, cell].astype(int)
        t_true = int(dd[r, de * 5 + st])
        sc = lambda lvl, t: SCORE[lvl - 1, st, 0, v][t]
        hon = S["hon"][i].astype(int)
        my_q = bool(owner[r, st * 13 + 2] == 2)
        rep = reply_of(hon, my_q, st)
        hon_t = ((owner[r, [s * 13 + j for s in range(4) for j in range(4)]] - 2) % 4)[None]
        rep_t = int(reply_of(hon_t, my_q, st)[0])
        # odds now: slam over game by average IMPs on the drawn deals
        now_slam = imps(sc(6, tr) - sc(lo, tr)).mean() > 0
        res_now = sc(6, t_true) if now_slam else sc(lo, t_true)
        # after the reply: 5 or 6 by the deals drawn for that reply
        b = rep == rep_t
        if b.sum() == 0:
            slam_after = now_slam
        else:
            slam_after = imps(sc(6, tr[b]) - sc(5, tr[b])).mean() > 0
        res_ask = sc(6, t_true) if slam_after else sc(5, t_true)
        # the same reply as free information: game still available below it
        slam_free = (imps(sc(6, tr[b]) - sc(lo, tr[b])).mean() > 0) if b.sum() else now_slam
        res_free = sc(6, t_true) if slam_free else sc(lo, t_true)
        # 5 of the suit instead of game when the choice stays game: the cost of the room alone
        res_room = sc(6, t_true) if now_slam else sc(5, t_true)
        # ask only when the drawn deals say asking beats bidding now (what hypothesis.py recommends)
        lvl_now = 6 if now_slam else lo
        pred = 0.0
        for j in range(4):
            bj = rep == j
            if bj.any():
                sj = imps(sc(6, tr[bj]) - sc(5, tr[bj])).mean() > 0
                pred += (imps(sc(6 if sj else 5, tr[bj]) - sc(lvl_now, tr[bj]))).sum() / len(tr)
        res_sel = res_ask if pred > 0 else res_now
        res_dd = max(sc(6, t_true), sc(lo, t_true))
        p_rep = np.bincount(rep, minlength=4) / len(rep)
        said += p_rep * count[r]
        true_hist[rep_t] += count[r]
        rows.append({"deal": int(S["deal_id"][r]), "w": count[r], "ask": float(imps(res_ask - res_now)),
                     "free": float(imps(res_free - res_now)), "room": float(imps(res_room - res_now)),
                     "sel": float(imps(res_sel - res_now)), "asked": float(pred > 0), "pred": float(pred) if pred > 0 else 0.0,
                     "dd": float(imps(res_dd - res_now)), "key": float(x["loss_now"] - x["loss_key"]),
                     "p_true_reply": float(p_rep[rep_t]), "now_slam": bool(now_slam), "slam_after": bool(slam_after)})
    if not rows:
        print("no decisions")
        return
    w = np.array([x["w"] for x in rows])
    g = np.array([x["deal"] for x in rows])
    n_deals = len(np.unique(S["deal_id"]))
    print(f"{a.tag}: {len(rows)} game-or-slam decisions in a suit with room for 4NT, on {len(np.unique(g))} of {n_deals} boards "
          f"({w.sum():.0f} tables); IMPs a decision against choosing by the odds now, double dummy, +- twice the standard error")
    for k, name in (("ask", "RKCB, then 5 or 6 by the odds after the reply"),
                    ("free", "  the reply known, game still possible"),
                    ("room", "  no reply, 5 instead of game (cost of the room)"),
                    ("sel", "RKCB only where the drawn deals say it pays"),
                    ("pred", "  what the drawn deals said it would gain"),
                    ("key", "keycards.py's key set known for certain"),
                    ("dd", "the right contract known (ceiling)")):
        m, se = wmean(np.array([x[k] for x in rows]), w, g)
        print(f"  {name:<50}{m:+.2f} +-{se:.2f}")
    print(f"  asked (drawn deals say it pays) on {wmean(np.array([x['asked'] for x in rows]), w, g)[0]:.0%} of decisions")
    ns = np.array([x["now_slam"] for x in rows])
    sa = np.array([x["slam_after"] for x in rows])
    print(f"  odds now bid slam on {wmean(ns.astype(float), w, g)[0]:.0%}; after the reply on {wmean(sa.astype(float), w, g)[0]:.0%}")
    print("  replies, said by the model -> true (weighted by tables): " +
          ", ".join(f"{REPLIES[j]} {said[j] / w.sum():.0%} -> {true_hist[j] / w.sum():.0%}" for j in range(4)))
    pt = np.array([x["p_true_reply"] for x in rows])
    print(f"  model's probability of the reply that came: mean {wmean(pt, w, g)[0]:.0%}; under 10% on {wmean((pt < 0.1).astype(float), w, g)[0]:.0%}")


if __name__ == "__main__":
    main()
