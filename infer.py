"""Use a trained auction model: one hand + the auction so far ->
  - double-dummy trick probabilities for every declarer and strain, expected DD score of any contract
  - par score distribution
  - sets (suit lengths and A/K/Q/J of the three hidden hands) with probabilities:
    ranked full sets, probability of any partial set, and the sets that matter for a choice of contracts.

  python infer.py --run /data/runs/v1 --hand AK52.KQ9742.Q.K2 --seat S --dealer S --vul E_W --auction "1H p 1S p"
  python infer.py --run /data/runs/v1 --selftest
"""
import argparse
from collections import Counter

import numpy as np
import torch

from model import AuctionModel, N_SLOTS, sample_sets, set_logits, set_masks, set_values, view_inputs
from scoring import SCORE, STRAINS

SEATS = "NESW"
RANKS = "AKQJT98765432"
HONORS = [s + r for s in "SHDC" for r in "AKQJ"]
VUL_BITS = {"None": 0, "N_S": 1, "NS": 1, "E_W": 2, "EW": 2, "Both": 3, "All": 3}
REL = ["me", "LHO", "partner", "RHO"]


def call_token(c):
    c = c.strip().upper()
    if c in ("P", "PASS"):
        return 1
    if c in ("D", "X", "DBL"):
        return 2
    if c in ("R", "XX", "RDBL"):
        return 3
    return 4 + (int(c[0]) - 1) * 5 + "CDHSN".index(c[1])


class Position:
    """The model's view from one seat after some calls."""

    def __init__(self, pred, hand, seat, dealer, vul, auction):
        self.pred = pred
        m, dev = pred.model, pred.dev
        self.seat = SEATS.index(seat)
        self.vul_bits = VUL_BITS[vul]
        owner = np.full(52, (self.seat + 1) % 4, np.uint8)
        suits = hand.split(".")
        assert len(suits) == 4 and sum(len(s) for s in suits) == 13, "hand must be 13 cards as S.H.D.C"
        for si, s in enumerate(suits):
            for ch in s.upper():
                owner[si * 13 + RANKS.index(ch)] = self.seat
        calls = [call_token(c) for c in auction]
        ct = torch.tensor([calls], dtype=torch.long, device=dev).view(1, len(calls))
        feats, callers, lens, hon = view_inputs(
            torch.tensor(owner[None], device=dev), torch.tensor([SEATS.index(dealer)], device=dev),
            torch.tensor([self.vul_bits], device=dev), ct)
        v = self.seat
        with torch.no_grad():
            h, f = m.encode(feats[v:v + 1], ct, callers[v:v + 1])
            self.h, self.f = h[:, len(calls)], f
            self.dd = m.dd_head(self.h).float().view(4, 5, 14).softmax(-1).cpu().numpy()      # rel declarer, strain, tricks
            self.par = m.par_head(self.h).float().softmax(-1)[0].cpu().numpy()
        self.mylen = lens[v, :, 0]
        self.myhon = hon[v] == 0
        self._samples = None

    # ---- double dummy and par -----------------------------------------------------------------
    def seat_name(self, rel):
        return SEATS[(self.seat + rel) % 4]

    def side_vul(self, rel):
        ns = (self.seat + rel) % 2 == 0
        return bool(self.vul_bits & (1 if ns else 2))

    def expected_tricks(self):
        """{declarer seat: {strain: expected tricks}}"""
        e = (self.dd * np.arange(14)).sum(-1)
        return {self.seat_name(r): {STRAINS[s]: float(e[r, s]) for s in range(5)} for r in range(4)}

    def _contract(self, contract, declarer):
        level, strain = int(contract[0]), STRAINS.index(contract[1].upper())
        dbl = 2 if contract.lower().endswith("xx") else 1 if contract.lower().endswith("x") else 0
        rel = (SEATS.index(declarer) - self.seat) % 4
        sign = 1.0 if rel % 2 == 0 else -1.0
        return rel, strain, sign * SCORE[level - 1, strain, dbl, int(self.side_vul(rel))], level + 6

    def contract(self, contract, declarer, dd=None):
        """Expected DD score for my side and the chance the contract makes. contract like '4S', '3Nx'."""
        rel, strain, sc, need = self._contract(contract, declarer)
        p = self.dd[rel, strain] if dd is None else dd[..., rel, strain, :]
        made = p[..., need:].sum(-1)
        return (p * sc).sum(-1), (made if rel % 2 == 0 else 1 - made)

    def par_summary(self, top=5):
        vals = np.array(self.pred.par_values)
        order = np.argsort(-self.par)[:top]
        return float((self.par * vals).sum()), [(int(vals[i]), float(self.par[i])) for i in order]

    # ---- sets ---------------------------------------------------------------------------------
    def samples(self, n=4096):
        """n sets drawn from the model: lens (n, suit, 3 hidden seats), hon (n,16) rel seat of each honor,
        logp (n,) model log probability of the set, dd (n,4,5,14) trick probabilities given the set."""
        if self._samples is None or self._samples["lens"].shape[0] != n:
            m = self.pred.model
            h, f = self.h.expand(n, -1), self.f.expand(n, -1)
            values, lens, hon, logp, st = sample_sets(m, h, f, self.mylen[None].expand(n, -1), self.myhon[None].expand(n, -1))
            with torch.no_grad():
                dd = m.ddset_head(st).float().view(n, 4, 5, 14).softmax(-1)
            self._samples = {"lens": lens.cpu().numpy(), "hon": hon.cpu().numpy(), "logp": logp.cpu().numpy(),
                             "dd": dd.cpu().numpy(), "values": values, "state": st}
        return self._samples

    def describe(self, lens, hon):
        parts = []
        for r in (1, 2, 3):
            hs = " ".join(HONORS[j] for j in range(16) if hon[j] == r) or "no honors"
            parts.append(f"{REL[r]} ({self.seat_name(r)}) {'-'.join(str(int(x)) for x in lens[:, r - 1])}: {hs}")
        return "; ".join(parts)

    def top_sets(self, n=4096, top=10):
        """Most likely full sets among n samples, each with the model's own probability of that set."""
        S = self.samples(n)
        seen = {}
        for i in range(n):
            key = (S["lens"][i].tobytes(), S["hon"][i].tobytes())
            if key not in seen:
                seen[key] = i
        idx = sorted(seen.values(), key=lambda i: -S["logp"][i])[:top]
        return [(float(np.exp(S["logp"][i])), self.describe(S["lens"][i], S["hon"][i])) for i in idx]

    def probability(self, honors=None, lengths=None, n=4096):
        """Probability of a partial set. honors: {'SA': 'LHO', 'SK': 'partner'}; lengths: {('LHO','S'): 5,
        ('RHO','C'): (4, 13)} (a number or an inclusive range). Seats: LHO, partner, RHO, or N/E/S/W."""
        S = self.samples(n)
        ok = np.ones(n, bool)
        for card, who in (honors or {}).items():
            ok &= S["hon"][:, HONORS.index(card.upper())] == self._rel(who)
        for (who, suit), rng in (lengths or {}).items():
            lo, hi = (rng, rng) if isinstance(rng, int) else rng
            x = S["lens"][:, "SHDC".index(suit.upper()), self._rel(who) - 1]
            ok &= (x >= lo) & (x <= hi)
        return float(ok.mean())

    def _conditions(self, conds):
        """Merge split conditions into one line per hidden seat, e.g. 'partner: 4+ S, HA, no CA'."""
        rng, has, lacks = {}, {}, {}
        for c in conds:
            if c[0] == "len":
                _, r, s, lo, hi = c
                a, b = rng.get((r, s), (0, 13))
                rng[(r, s)] = (max(a, lo), min(b, hi))
            elif c[3]:
                has[c[1]] = c[2]
            else:
                lacks.setdefault(c[1], set()).add(c[2])
        for j, rs in lacks.items():
            if j not in has and len(rs) == 2:
                has[j] = ({1, 2, 3} - rs).pop()
        out = []
        for r in (1, 2, 3):
            items = []
            for s in range(4):
                if (r, s) in rng:
                    lo, hi = rng[(r, s)]
                    items.append(f"{lo}+ {'SHDC'[s]}" if hi == 13 else f"at most {hi} {'SHDC'[s]}" if lo == 0 else
                                 f"exactly {lo} {'SHDC'[s]}" if lo == hi else f"{lo}-{hi} {'SHDC'[s]}")
            items += [HONORS[j] for j in range(16) if has.get(j) == r]
            items += ["no " + HONORS[j] for j in range(16) if j not in has and r in lacks.get(j, ())]
            if items:
                out.append(f"{REL[r]} ({self.seat_name(r)}): " + ", ".join(items))
        return out

    def _rel(self, who):
        if who in SEATS:
            r = (SEATS.index(who) - self.seat) % 4
        else:
            r = [x.lower() for x in REL].index(who.lower())
        assert r != 0, "that is my own seat"
        return r

    def marginals(self, n=4096):
        """Per honor: probability at LHO / partner / RHO. Per hidden seat and suit: expected length."""
        S = self.samples(n)
        hon = {HONORS[j]: {REL[r]: float((S["hon"][:, j] == r).mean()) for r in (1, 2, 3)}
               for j in range(16) if not bool(self.myhon[j])}
        ln = {REL[r]: {"SHDC"[s]: float(S["lens"][:, s, r - 1].mean()) for s in range(4)} for r in (1, 2, 3)}
        return hon, ln

    def significant_sets(self, candidates, n=4096, depth=3, min_prob=0.04):
        """Sets that matter for choosing between contracts. candidates: [('4S','S'), ('3N','N'), ...].
        Splits the sampled sets on the honor locations and suit lengths that change the candidates' expected
        DD scores the most. Returns leaves: (probability, conditions, {candidate: (score, chance of making)})."""
        S = self.samples(n)
        Y = np.stack([self.contract(c, d, S["dd"])[0] for c, d in candidates], axis=1)       # (n, C) my side's score
        M = np.stack([self.contract(c, d, S["dd"])[1] for c, d in candidates], axis=1)
        names, cols = [], []                     # names: (kind, ...) for the feature being true / false
        for j in range(16):
            if bool(self.myhon[j]):
                continue
            for r in (1, 2, 3):
                names.append((("hon", j, r, True), ("hon", j, r, False)))
                cols.append(S["hon"][:, j] == r)
        for r in (1, 2, 3):
            for s in range(4):
                x = S["lens"][:, s, r - 1]
                for k in range(1, 9):
                    names.append((("len", r, s, k, 13), ("len", r, s, 0, k - 1)))
                    cols.append(x >= k)
        X = np.stack(cols, axis=1)
        leaves = []

        def grow(rows, conds, d):
            best = None
            if d < depth and rows.sum() >= 2 * min_prob * n:
                y = Y[rows]
                x = X[rows]
                n1 = x.sum(0).astype(np.float64)
                n0 = len(y) - n1
                ok = (n1 >= min_prob * n) & (n0 >= min_prob * n)
                if ok.any():
                    s1 = x.T.astype(np.float64) @ y                                          # (F, C)
                    mean1 = s1 / np.maximum(n1, 1)[:, None]
                    mean0 = (y.sum(0)[None, :] - s1) / np.maximum(n0, 1)[:, None]
                    gain = (n1 * n0 / len(y))[:, None] * (mean1 - mean0) ** 2                 # between-group sum of squares
                    gain = np.where(ok, gain.sum(1), -1.0)
                    best = int(gain.argmax())
                    if gain[best] <= len(y) * 25.0:                                           # under 5 points of score: not worth a split
                        best = None
            if best is None:
                leaves.append((float(rows.mean()), self._conditions(conds),
                               {f"{c} by {dd}": (float(Y[rows, i].mean()), float(M[rows, i].mean()))
                                for i, (c, dd) in enumerate(candidates)}))
                return
            grow(rows & X[:, best], conds + [names[best][0]], d + 1)
            grow(rows & ~X[:, best], conds + [names[best][1]], d + 1)

        grow(np.ones(n, bool), [], 0)
        return sorted(leaves, key=lambda t: -t[0])


class Predictor:
    def __init__(self, run, device=None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.dev.type == "cpu":
            torch.set_num_threads(8)
        ck = torch.load(f"{run}/model.pt", map_location=self.dev, weights_only=False)
        self.model = AuctionModel(**ck["cfg"]).to(self.dev).eval()
        self.model.load_state_dict(ck["model"])
        self.par_values = ck["par_values"]
        self.step = ck.get("step")

    def position(self, hand, seat, dealer, vul, auction):
        return Position(self, hand, seat, dealer, vul, auction if isinstance(auction, list) else auction.split())


def show(pos, candidates=None, n=4096):
    print("Expected double-dummy tricks (declarer by strain):")
    et = pos.expected_tricks()
    print("      " + "".join(f"{s:>7}" for s in STRAINS))
    for d in SEATS:
        print(f"  {d}   " + "".join(f"{et[d][s]:7.2f}" for s in STRAINS))
    mean, top = pos.par_summary()
    me = SEATS[pos.seat]
    print(f"Par for {me}'s side: mean {mean:+.0f}; most likely " + ", ".join(f"{v:+d} ({p:.0%})" for v, p in top))
    if candidates is None:
        e = (pos.dd * np.arange(14)).sum(-1)
        candidates = []
        for s in np.argsort(-np.maximum(e[0], e[2]))[:3]:
            r = 0 if e[0, s] >= e[2, s] else 2
            best = max(range(1, 8), key=lambda l: pos.contract(f"{l}{STRAINS[s]}", pos.seat_name(r))[0])
            candidates.append((f"{best}{STRAINS[s]}", pos.seat_name(r)))
    print("Contracts (expected DD score for my side, chance of making):")
    for c, d in candidates:
        sc, mk = pos.contract(c, d)
        print(f"  {c} by {d}: {sc:+.0f}, makes {mk:.0%}")
    hon, ln = pos.marginals(n)
    print("Where the missing honors are (LHO / partner / RHO):")
    print("  " + "  ".join(f"{k} {v['LHO']:.0%}/{v['partner']:.0%}/{v['RHO']:.0%}" for k, v in hon.items()))
    print("Expected suit lengths S-H-D-C: " + "; ".join(
        f"{r} {'-'.join(f'{ln[r][s]:.1f}' for s in 'SHDC')}" for r in ("LHO", "partner", "RHO")))
    print(f"Most likely full sets ({n} samples):")
    for p, d in pos.top_sets(n, 5):
        print(f"  1 in {1/p:,.0f}  {d}")
    print("Sets that matter for these contracts:")
    for p, conds, sc in pos.significant_sets(candidates, n):
        print(f"  {p:5.1%}  {'; '.join(conds) or '(no split)'}")
        print("         " + "; ".join(f"{k}: {v[0]:+.0f}, makes {v[1]:.0%}" for k, v in sc.items()))


def selftest(pred, n=2000):
    """Sampled sets must be legal deals of the 39 hidden cards."""
    pos = pred.position("AK52.KQ9742.Q.K2", "S", "S", "E_W", "1H p 1S p")
    S = pos.samples(n)
    lens, hon = S["lens"], S["hon"]
    mylen = pos.mylen.cpu().numpy()
    assert (lens.sum(1) == 13).all(), "a hidden hand does not have 13 cards"
    assert (lens.sum(2) == 13 - mylen[None, :]).all(), "a suit does not add up"
    myhon = pos.myhon.cpu().numpy()
    assert ((hon == 0) == myhon[None, :]).all(), "own honors misplaced"
    for s in range(4):
        for r in (1, 2, 3):
            assert ((hon[:, s * 4:s * 4 + 4] == r).sum(1) <= lens[:, s, r - 1]).all(), "more honors than cards in a suit"
    # the step-by-step sampler must agree with the training-time (teacher-forced) path
    m, dev = pred.model, pred.dev
    lens4 = torch.cat([pos.mylen[None, :, None].expand(n, -1, -1), torch.tensor(lens, device=dev)], dim=2)
    hon_t = torch.tensor(hon, device=dev)
    values = set_values(lens4, hon_t)
    assert (values == S["values"]).all(), "value tokens differ"
    len_mask, hon_mask = set_masks(lens4, hon_t)
    with torch.no_grad():
        st = m.decode(pos.h.expand(n, -1), pos.f.expand(n, -1), values)
        ll, hl = set_logits(m, st, len_mask, hon_mask)
        lp = -torch.nn.functional.cross_entropy(ll.reshape(-1, 14), values[:, :6].reshape(-1), reduction="none").view(n, 6).sum(1)
        hv = hon_t > 0
        lp -= (torch.nn.functional.cross_entropy(hl.reshape(-1, 3), (hon_t - 1).clamp(min=0).reshape(-1),
                                                 reduction="none").view(n, 16) * hv).sum(1)
    d_lp = float((lp.cpu() - torch.tensor(S["logp"])).abs().max())
    d_st = float((st[:, N_SLOTS] - S["state"]).abs().max())
    assert d_lp < 1e-2 and d_st < 1e-2, (d_lp, d_st)
    distinct = len(Counter(map(bytes, np.concatenate([lens.reshape(n, -1), hon], 1))))
    print(f"selftest ok: {n} sampled sets are legal ({distinct} distinct); sampler and training path agree "
          f"(log-probability difference {d_lp:.1e}, state difference {d_st:.1e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--hand")
    ap.add_argument("--seat", default="S")
    ap.add_argument("--dealer", default="N")
    ap.add_argument("--vul", default="None")
    ap.add_argument("--auction", default="")
    ap.add_argument("--contracts", default="", help="e.g. '4S:S 3N:N 5Cx:W'")
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    P = Predictor(a.run, a.device)
    print(f"model at step {P.step} on {P.dev}")
    if a.selftest:
        selftest(P)
    if a.hand:
        cands = [tuple(x.split(":")) for x in a.contracts.split()] or None
        show(P.position(a.hand, a.seat, a.dealer, a.vul, a.auction), cands, a.samples)
