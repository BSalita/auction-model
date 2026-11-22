"""Hypotheses and asking bids: "if partner has DA and CK, slam makes" -> how likely is that, given the auction, and
which contract or which asking bid is worth the most.

From one hand and the auction so far, the auction model draws sets (suit lengths and A K Q J of the three hidden
hands) consistent with all the calls seen. Each set is completed to a deal (cards below the jack at random within
the lengths) and solved double dummy for the candidate contracts. Then:

  --if H     P(H); expected result and chance of making of every contract on all deals, when H is true and when it
             is not; the best contract each way; the break-even P(H); what knowing H for certain is worth over
             choosing by the odds now (and, as a ceiling, what knowing the whole deal is worth).
  --ask A    an asking bid splits the deals by partner's reply. For each reply: its probability, P(H) after it, the
             best contract still reachable, and what asking then placing the contract is worth over bidding now.
             A: rkcb:S (1430: 5C 1/4, 5D 0/3, 5H 2 without the queen, 5S 2 with it), rkcb0314:S, aces:S
             (4NT: 5C 0/4, 5D 1, 5H 2, 5S 3), or any hypothesis as an ideal yes/no question, optionally with the
             cheapest contract left after the answer: "partner: DA CK @5S".

Hypotheses (one or more terms, all must hold; "or" between alternatives; a seat prefix holds until the next one;
the default seat is partner):
  partner: DA CK           honors (suit then rank or rank then suit)       no SA / not SA     LHO: HK
  3+ S   0-1 H   exactly 4 D   at most 2 C   void S   singleton H          suit lengths
  2+ aces   1 king   0-1 queens   3 keycards S   2+ kc in H                 counts (keycards: 4 aces + trump king)
  12+ hcp   15-17 points                                                    high-card points
  e.g. "partner: DA CK; RHO: no HA"   "partner: 2 keycards S and SQ or partner: 3+ keycards S"

Scoring: --scoring imps (default) chooses a contract by its average IMPs against each of the other candidates (for two
candidates, the IMPs one gains over the other); mp by the average share of a matchpoint against each; total by points.

Everything is double dummy (perfect play by all four). The model's odds are honest on BBO deals, a little
overconfident on Brill deals, and slam odds from sampled deals ran high on Brill boards (said 40%, true 31-34%:
keycards.py, README "Key cards").

Run where the model and the solver are (the BEN image, PyTorch in /var/lib/ben/torch-pkgs):

  export PYTHONPATH=/var/lib/ben/torch-pkgs
  python hypothesis.py --run /data/runs/v3-brill --hand AKJ873.KQ4.K2.A5 --dealer S --vul None \\
      --auction "1S p 3S p" --contracts "4S 6S 6N" --if "partner: DA CK" --ask rkcb:S --ask "partner: DA CK @5S"

--solver model uses the model's own trick odds given each set instead of solving (instant; exact tricks 68%).
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import time

import numpy as np

from scoring import SCORE

SEATS = "NESW"
SUITS = "SHDC"
STRAINS = "SHDCN"
RANKS = "AKQJT98765432"
HONORS = [s + r for s in SUITS for r in "AKQJ"]
REL = ["me", "LHO", "partner", "RHO"]
HCP = np.array([4, 3, 2, 1] * 4)
IMP_T = np.array([20, 50, 90, 130, 170, 220, 270, 320, 370, 430, 500, 600, 750, 900, 1100, 1300, 1500, 1750, 2000,
                  2250, 2500, 3000, 3500, 4000])


def imps(x):
    return np.sign(x) * np.searchsorted(IMP_T, np.abs(x), side="right")


def call_rank(level, strain):
    """Bidding order of a call; strain index into STRAINS (S H D C N)."""
    return (level - 1) * 5 + "CDHSN".index(STRAINS[strain])


# ---------------------------------------------------------------------------------------------- hypotheses

_SEAT = re.compile(r"\s*(partner|pard|lho|rho|me)\b\s*(?::|\bhas\b|\bholds\b)?", re.I)
_SKIP = re.compile(r"(\s+|,|;|&|\band\b|\bhas\b|\bholds\b|\bwith\b)+", re.I)
_NEG = re.compile(r"(no|not)\s+|[-!]\s*", re.I)
_RANGE = r"(?:(?:at\s*most|<=)\s*(?P<amx>\d+)|exactly\s*(?P<ex>\d+)|(?P<lo>\d+)\s*(?:(?P<plus>\+)|-\s*(?P<hi>\d+))?)\s*"
_ITEMS = [
    ("hon", re.compile(r"(?P<a>[SHDC][AKQJ]|[AKQJ][SHDC])\b", re.I)),
    ("kc", re.compile(_RANGE + r"(?:key\s*cards?|kc)\s*(?:in\s*)?(?P<suit>[SHDC])\b", re.I)),
    ("count", re.compile(_RANGE + r"(?P<rank>aces?|kings?|queens?|jacks?)\b", re.I)),
    ("hcp", re.compile(_RANGE + r"(?:hcp|points|pts)\b", re.I)),
    ("short", re.compile(r"(?P<kind>void|singleton|stiff|doubleton)\s*(?:in\s*)?(?P<suit>[SHDC])\b", re.I)),
    ("len", re.compile(_RANGE + r"(?P<suit>[SHDC])\b(?:\s*cards?)?", re.I)),
]


def _range(m, top):
    g = m.groupdict()
    if g.get("amx"):
        return 0, int(g["amx"])
    if g.get("ex"):
        return int(g["ex"]), int(g["ex"])
    lo = int(g["lo"])
    return (lo, top) if g.get("plus") else (lo, int(g["hi"])) if g.get("hi") else (lo, lo)


def _rng_text(lo, hi, top):
    return f"{lo}+" if hi >= top else f"at most {hi}" if lo == 0 else str(lo) if lo == hi else f"{lo}-{hi}"


class Hypothesis:
    """A condition on the three hidden hands. mask(lens, hon) -> bool (n,). lens (n,4,3) for LHO/partner/RHO,
    hon (n,16) relative seat of each honor (0 = mine)."""

    def __init__(self, text, mylen, myhon):
        self.text = text.strip()
        self.mylen, self.myhon = np.asarray(mylen), np.asarray(myhon, bool)
        self.alts = [self._parse(a) for a in re.split(r"\s+or\s+|\|", self.text, flags=re.I) if a.strip()]
        assert self.alts, "empty hypothesis"
        self.desc = " or ".join(self._join([d for _, d in alt]) for alt in self.alts)

    @staticmethod
    def _join(ds):
        """'partner: DA', 'partner: CK', 'RHO: no HA' -> 'partner: DA CK; RHO: no HA'"""
        out, last = [], None
        for d in ds:
            neg = d.startswith("not (")
            who, _, item = (d[5:-1] if neg else d).partition(": ")
            item = ("no " + item) if neg and re.fullmatch(r"[SHDC][AKQJ]", item) else ("not " + item) if neg else item
            if who == last:
                out[-1] += (" " if re.fullmatch(r"(no )?[SHDC][AKQJ]", item) and re.search(r"[SHDC][AKQJ]$", out[-1]) else ", ") + item
            else:
                out.append(f"{who}: {item}")
                last = who
        return "; ".join(out)

    def _parse(self, s):
        terms, who, i = [], 2, 0
        while i < len(s):
            m = _SKIP.match(s, i)
            if m and m.end() > i:
                i = m.end()
                continue
            m = _SEAT.match(s, i)
            if m:
                w = m.group(1).lower()
                who = {"partner": 2, "pard": 2, "lho": 1, "rho": 3, "me": 0}[w]
                i = m.end()
                continue
            neg = False
            m = _NEG.match(s, i)
            if m:
                neg, i = True, m.end()
            for kind, rx in _ITEMS:
                m = rx.match(s, i)
                if m:
                    break
            else:
                raise ValueError(f"cannot read '{s[i:]}' in the hypothesis")
            i = m.end()
            terms.append(self._term(kind, m, who, neg))
        return terms

    def _term(self, kind, m, who, neg):
        name = REL[who]
        if kind == "hon":
            a = m.group("a").upper()
            card = a if a[0] in SUITS else a[1] + a[0]
            j = HONORS.index(card)
            f = lambda L, H, j=j: H[:, j] == who
            d = f"{name}: {card}"
        elif kind == "len":
            s = SUITS.index(m.group("suit").upper())
            lo, hi = _range(m, 13)
            f = lambda L, H, s=s, lo=lo, hi=hi: self._len(L, s, who, lo, hi)
            d = f"{name}: {_rng_text(lo, hi, 13)} {SUITS[s]}"
        elif kind == "short":
            s = SUITS.index(m.group("suit").upper())
            k = m.group("kind").lower()
            n = {"void": 0, "singleton": 1, "stiff": 1, "doubleton": 2}[k]
            f = lambda L, H, s=s, n=n: self._len(L, s, who, n, n)
            d = f"{name}: {k} {SUITS[s]}"
        elif kind == "count":
            rk = "akqj".index(m.group("rank")[0].lower())
            lo, hi = _range(m, 4)
            cols = [s * 4 + rk for s in range(4)]
            f = lambda L, H, cols=cols, lo=lo, hi=hi: self._between(self._held(H, cols, who).sum(1), lo, hi)
            d = f"{name}: {_rng_text(lo, hi, 4)} {m.group('rank').lower()}"
        elif kind == "kc":
            t = SUITS.index(m.group("suit").upper())
            lo, hi = _range(m, 5)
            cols = [0, 4, 8, 12, t * 4 + 1]
            f = lambda L, H, cols=cols, lo=lo, hi=hi: self._between(self._held(H, cols, who).sum(1), lo, hi)
            d = f"{name}: {_rng_text(lo, hi, 5)} keycards in {SUITS[t]}"
        else:
            lo, hi = _range(m, 40)
            f = lambda L, H, lo=lo, hi=hi: self._between((self._held(H, list(range(16)), who) * HCP).sum(1), lo, hi)
            d = f"{name}: {_rng_text(lo, hi, 40)} hcp"
        if neg:
            return (lambda L, H, f=f: ~f(L, H)), "not (" + d + ")"
        return f, d

    def _held(self, H, cols, who):
        if who == 0:
            return np.broadcast_to(self.myhon[cols], (H.shape[0], len(cols))).astype(int)
        return (H[:, cols] == who).astype(int)

    def _len(self, L, s, who, lo, hi):
        x = np.full(L.shape[0], self.mylen[s]) if who == 0 else L[:, s, who - 1]
        return self._between(x, lo, hi)

    @staticmethod
    def _between(x, lo, hi):
        return (x >= lo) & (x <= hi)

    def mask(self, lens, hon):
        out = np.zeros(lens.shape[0], bool)
        for alt in self.alts:
            m = np.ones(lens.shape[0], bool)
            for f, _ in alt:
                m &= f(lens, hon)
            out |= m
        return out


# ---------------------------------------------------------------------------------------------- the query

def _solve_init():
    import keycards
    keycards.solve_init()


def _solve(task):
    import keycards
    return keycards.solve_state(task)


class Query:
    """Sets drawn for one position, completed and solved for the candidate contracts.

    contracts: ["4S", "6S:S", "6N:partner", "5Dx:W"]; declarer N/E/S/W or me/partner/LHO/RHO (default me)."""

    def __init__(self, pos, hand, contracts, n=1000, solver="dds", workers=16, seed=1, scoring="imps"):
        self.pos, self.n, self.scoring = pos, n, scoring
        self.mine = np.zeros(52, bool)
        for si, s in enumerate(hand.split(".")):
            for ch in s.upper():
                self.mine[si * 13 + RANKS.index(ch)] = True
        self.mylen = pos.mylen.cpu().numpy() if hasattr(pos.mylen, "cpu") else np.asarray(pos.mylen)
        self.myhon = pos.myhon.cpu().numpy() if hasattr(pos.myhon, "cpu") else np.asarray(pos.myhon)
        S = pos.samples(n)
        self.lens, self.hon = S["lens"].astype(int), S["hon"].astype(int)
        self.contracts = []
        for c in contracts:
            self.add_contract(c)
        self.solver, self.workers, self.seed = solver, workers, seed
        self._tricks = {}                                       # (rel declarer, strain) -> (n,) tricks
        self._dd = S["dd"]
        self._solve_cells()

    # ---- contracts and tricks
    def parse_contract(self, c):
        c = c.strip()
        txt, _, who = c.partition(":")
        m = re.fullmatch(r"([1-7])([SHDCN])(XX|X)?", txt.strip(), re.I)
        assert m, f"contract '{c}': like 4S, 6N:partner, 5Dx:W"
        level, strain = int(m.group(1)), STRAINS.index(m.group(2).upper())
        dbl = {None: 0, "X": 1, "XX": 2}[m.group(3).upper() if m.group(3) else None]
        who = who.strip() or "me"
        if who.upper() in SEATS and len(who) == 1:
            rel = (SEATS.index(who.upper()) - self.pos.seat) % 4
        else:
            rel = [x.lower() for x in REL].index(who.lower())
        label = f"{level}{STRAINS[strain]}{'' if dbl == 0 else 'x' * dbl} by {self.pos.seat_name(rel)}"
        return {"label": label, "level": level, "strain": strain, "dbl": dbl, "rel": rel}

    def add_contract(self, c):
        k = self.parse_contract(c) if isinstance(c, str) else c
        for i, x in enumerate(self.contracts):
            if x["label"] == k["label"]:
                return i
        self.contracts.append(k)
        if hasattr(self, "_tricks"):
            self._solve_cells()
        return len(self.contracts) - 1

    def _solve_cells(self):
        need = sorted({(x["rel"], x["strain"]) for x in self.contracts} - set(self._tricks))
        if not need:
            return
        t0 = time.time()
        if self.solver == "model":
            rng = np.random.default_rng(self.seed)
            for rel, st in need:
                p = self._dd[:, rel, st, :]
                cum = p.cumsum(1)
                u = rng.random((self.n, 1)) * cum[:, -1:]
                self._tricks[(rel, st)] = (u > cum).sum(1)
        else:
            # keycards.solve_state seats me South (absolute 2) and honors of relative seat r at (2 + r) % 4
            cells = [((2 + rel) % 4, st) for rel, st in need]
            step = max(1, -(-self.n // (self.workers * 4)))
            tasks = [(i, self.mine, self.lens[i:i + step], self.hon[i:i + step], cells, self.seed * 100003 + i)
                     for i in range(0, self.n, step)]
            out = np.zeros((self.n, len(cells)), np.uint8)
            if self.workers <= 1:
                _solve_init()
                res = map(_solve, tasks)
                for i, o in res:
                    out[i:i + len(o)] = o
            else:
                with mp.get_context("spawn").Pool(min(self.workers, len(tasks)), _solve_init) as pool:
                    for i, o in pool.imap_unordered(_solve, tasks):
                        out[i:i + len(o)] = o
            for c, key in enumerate(need):
                self._tricks[key] = out[:, c].astype(int)
        self.solve_seconds = time.time() - t0

    def tricks(self, k):
        x = self.contracts[k]
        return self._tricks[(x["rel"], x["strain"])]

    def score(self, k):
        """(n,) score for my side of contract k on each deal."""
        x = self.contracts[k]
        sign = 1.0 if x["rel"] % 2 == 0 else -1.0
        vul = int(self.pos.side_vul(x["rel"]))
        return sign * SCORE[x["level"] - 1, x["strain"], x["dbl"], vul][self.tricks(k)]

    def makes(self, k):
        x = self.contracts[k]
        m = self.tricks(k) >= x["level"] + 6
        return m if x["rel"] % 2 == 0 else ~m

    # ---- choosing
    def _pair(self, a, b):
        """(n,) what contract a gains over b on each deal, in the scoring's unit."""
        d = self.score(a) - self.score(b)
        if self.scoring == "imps":
            return imps(d).astype(float)
        if self.scoring == "mp":
            return np.sign(d) / 2.0
        return d

    def value(self, k, allowed, rows):
        """Average over deals in rows of what k gains against each other allowed contract (0 if k is alone)."""
        others = [j for j in allowed if j != k]
        if not others or rows.sum() == 0:
            return 0.0
        return float(np.mean([self._pair(k, j)[rows].mean() for j in others]))

    def best(self, allowed, rows):
        return max(allowed, key=lambda k: (self.value(k, allowed, rows), self.score(k)[rows].mean() if rows.any() else 0))

    def gain(self, choice, ref):
        """choice (n,) contract index per deal against the fixed contract ref: mean and twice its standard error."""
        g = np.zeros(self.n)
        for c in np.unique(choice):
            if c != ref:
                sel = choice == c
                g[sel] = self._pair(int(c), ref)[sel]
        return float(g.mean()), float(2 * g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0

    def unit(self):
        return {"imps": "IMPs", "mp": "matchpoints (share of a top)", "total": "points"}[self.scoring]

    # ---- reports
    def summary(self, k, rows):
        s = self.score(k)[rows]
        return {"ev": float(s.mean()) if len(s) else float("nan"),
                "makes": float(self.makes(k)[rows].mean()) if len(s) else float("nan")}

    def hypothesis(self, text):
        H = Hypothesis(text, self.mylen, self.myhon)
        m = H.mask(self.lens, self.hon)
        allr, n = np.ones(self.n, bool), self.n
        cand = list(range(len(self.contracts)))
        p = float(m.mean())
        now = self.best(cand, allr)
        b1 = self.best(cand, m) if m.any() else now
        b0 = self.best(cand, ~m) if (~m).any() else now
        choice = np.where(m, b1, b0)
        g_know, se_know = self.gain(choice, now)
        # ceiling: the best contract on each deal
        per = np.stack([self.score(k) for k in cand], 1)
        g_dd, se_dd = self.gain(per.argmax(1), now)
        res = {"hypothesis": H.desc, "p": p, "p_se": float(2 * np.sqrt(p * (1 - p) / n)), "n": n,
               "contracts": {self.contracts[k]["label"]: {"all": self.summary(k, allr), "if": self.summary(k, m),
                                                          "if_not": self.summary(k, ~m)} for k in cand},
               "best_now": self.contracts[now]["label"], "best_if": self.contracts[b1]["label"],
               "best_if_not": self.contracts[b0]["label"], "knowing_gain": g_know, "knowing_se": se_know,
               "dd_gain": g_dd, "dd_se": se_dd, "unit": self.unit()}
        # every other contract against the choice now: what it gains each way and the P(H) that makes it right
        res["pairs"] = []
        for k in cand:
            if k == now:
                continue
            d = self._pair(k, now)
            d1 = float(d[m].mean()) if m.any() else float("nan")
            d0 = float(d[~m].mean()) if (~m).any() else float("nan")
            be = -d0 / (d1 - d0) if np.isfinite(d1) and np.isfinite(d0) and d1 != d0 else float("nan")
            res["pairs"].append({"a": self.contracts[k]["label"], "b": self.contracts[now]["label"], "gain_if": d1,
                                 "gain_if_not": d0, "break_even": float(be)})
        return res

    def ask(self, spec, hyps=()):
        """Value of an asking bid. spec: rkcb:S, rkcb0314:S, aces:S, or a hypothesis with an optional '@5S' floor."""
        cand = list(range(len(self.contracts)))
        now = self.best(cand, np.ones(self.n, bool))
        kind, _, arg = spec.partition(":")
        kind = kind.strip().lower()
        if kind in ("rkcb", "rkcb1430", "1430", "rkcb0314", "0314", "aces", "blackwood") and re.fullmatch(r"\s*[SHDC]\s*", arg):
            t = SUITS.index(arg.strip().upper())
            held = (self.hon[:, [0, 4, 8, 12]] == 2).sum(1)
            if kind in ("aces", "blackwood"):
                name = f"Blackwood 4NT (aces), {SUITS[t]} agreed"
                steps = [("5C 0 or 4 aces", held % 4 == 0), ("5D 1 ace", held == 1), ("5H 2 aces", held == 2),
                         ("5S 3 aces", held == 3)]
            else:
                kc = held + (self.hon[:, t * 4 + 1] == 2)
                q = (self.hon[:, t * 4 + 2] == 2) | bool(self.myhon[t * 4 + 2])
                r14, r03 = (kc == 1) | (kc >= 4), (kc == 0) | (kc == 3)            # all 5 keycards go with 1 or 4
                if kind in ("rkcb0314", "0314"):
                    name, first, second = f"RKCB 0314 for {SUITS[t]}", ("5C 0 or 3 keycards", r03), ("5D 1 or 4 keycards", r14)
                else:
                    name, first, second = f"RKCB 1430 for {SUITS[t]}", ("5C 1 or 4 keycards", r14), ("5D 0 or 3 keycards", r03)
                steps = [first, second, ("5H 2 keycards, no queen", (kc == 2) & ~q), ("5S 2 keycards and the queen", (kc == 2) & q)]
            # sign-off in the agreed suit by whoever plays it among the candidates
            decl = next((x["rel"] for x in self.contracts if x["strain"] == t), 0)
            self.add_contract({"label": f"5{SUITS[t]} by {self.pos.seat_name(decl)}", "level": 5, "strain": t, "dbl": 0, "rel": decl})
            branches = []
            for i, (lab, rows) in enumerate(steps):
                reply = call_rank(5, STRAINS.index("CDHS"[i]))
                ok = [k for k in range(len(self.contracts)) if call_rank(self.contracts[k]["level"], self.contracts[k]["strain"]) >= reply
                      and not (self.contracts[k]["level"] == 5 and self.contracts[k]["strain"] == 4)]
                branches.append((lab, rows, ok))
        else:
            text, _, floor = spec.partition("@")
            H = Hypothesis(text, self.mylen, self.myhon)
            rows = H.mask(self.lens, self.hon)
            ok = list(range(len(self.contracts)))
            name = f"ideal question: {H.desc}"
            if floor.strip():
                fl = self.parse_contract(floor.strip())
                ok = [k for k in ok if call_rank(self.contracts[k]["level"], self.contracts[k]["strain"]) >= call_rank(fl["level"], fl["strain"])]
                if not any(self.contracts[k]["label"] == fl["label"] for k in ok):
                    ok.append(self.add_contract(fl))
                name += f", then {fl['label'][:2]} or higher"
            branches = [("yes", rows, ok), ("no", ~rows, ok)]
        hmasks = [(Hypothesis(h, self.mylen, self.myhon)) for h in hyps]
        choice = np.full(self.n, now)
        out = []
        for lab, rows, ok in branches:
            if not ok:
                raise ValueError(f"no candidate contract is reachable after {lab}")
            b = self.best(ok, rows) if rows.any() else ok[0]
            choice[rows] = b
            out.append({"reply": lab, "p": float(rows.mean()), "best": self.contracts[b]["label"], **self.summary(b, rows),
                        "p_hyp": [float(h.mask(self.lens, self.hon)[rows].mean()) if rows.any() else float("nan") for h in hmasks]})
        g, se = self.gain(choice, now)
        return {"ask": name, "best_now": self.contracts[now]["label"], "replies": out, "gain": g, "se": se,
                "hyps": [h.desc for h in hmasks], "unit": self.unit()}


# ---------------------------------------------------------------------------------------------- printing

def print_hypothesis(r):
    print(f"\nIf {r['hypothesis']}")
    print(f"  P = {r['p']:.0%} +-{r['p_se']:.0%} on {r['n']} deals drawn for this hand and auction")
    w = max(len(k) for k in r["contracts"]) + 2
    print(f"  {'':<{w}}{'all deals':>18}{'if so':>18}{'if not':>18}")
    print(f"  {'contract':<{w}}" + f"{'score  makes':>18}" * 3)
    for k, v in r["contracts"].items():
        print(f"  {k:<{w}}" + "".join(f"{v[c]['ev']:>+10.0f}{v[c]['makes']:>7.0%} " for c in ("all", "if", "if_not")))
    print(f"  best: by the odds now {r['best_now']}; if so {r['best_if']}; if not {r['best_if_not']}")
    for q in r["pairs"]:
        be = q["break_even"]
        if not np.isfinite(be):
            verdict = ""
        elif q["gain_if"] > q["gain_if_not"]:
            verdict = (f"; right when P > {be:.0%}" if 0 < be < 1 else "; right whatever P" if be <= 0 else "; never right")
        else:
            verdict = (f"; right when P < {be:.0%}" if 0 < be < 1 else "; never right" if be <= 0 else "; right whatever P")
        print(f"  {q['a']} instead of {q['b']}: {q['gain_if']:+.2f} {r['unit']} if so, {q['gain_if_not']:+.2f} if not{verdict}")
    print(f"  knowing for certain: {r['knowing_gain']:+.2f} +-{r['knowing_se']:.2f} {r['unit']} over choosing by the odds; "
          f"knowing the whole deal: {r['dd_gain']:+.2f} +-{r['dd_se']:.2f}")


def print_ask(r):
    print(f"\nAsk: {r['ask']}  (bidding now by the odds: {r['best_now']})")
    hy = "".join(f"{'P(' + h + ')':>{max(14, len(h) + 5)}}" for h in r["hyps"])
    print(f"  {'reply':<30}{'chance':>8}{'then':>14}{'score':>8}{'makes':>7}{hy}")
    for b in r["replies"]:
        if b["p"] == 0:
            print(f"  {b['reply']:<30}{'0%':>8}   (impossible with this hand)")
            continue
        hv = "".join(f"{x:>{max(14, len(h) + 5)}.0%}" for x, h in zip(b["p_hyp"], r["hyps"]))
        print(f"  {b['reply']:<30}{b['p']:>8.0%}{b['best']:>14}{b['ev']:>+8.0f}{b['makes']:>7.0%}{hv}")
    print(f"  asking, then placing the contract: {r['gain']:+.2f} +-{r['se']:.2f} {r['unit']} over bidding {r['best_now']} now")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="/data/runs/v3-brill")
    ap.add_argument("--hand", required=True, help="S.H.D.C, e.g. AKJ873.KQ4.K2.A5")
    ap.add_argument("--seat", default="S")
    ap.add_argument("--dealer", default="S")
    ap.add_argument("--vul", default="None", help="None, NS, EW, Both")
    ap.add_argument("--auction", default="", help="calls so far, e.g. '1S p 3S p'")
    ap.add_argument("--contracts", required=True, help="e.g. '4S 6S 6N' or '4S:S 6S:N 5Dx:W'")
    ap.add_argument("--if", dest="hyps", action="append", default=[], help="a hypothesis; may be given more than once")
    ap.add_argument("--ask", action="append", default=[], help="rkcb:S, rkcb0314:S, aces:S, or 'hypothesis @5S'")
    ap.add_argument("--scoring", default="imps", choices=["imps", "mp", "total"])
    ap.add_argument("--n", type=int, default=1000, help="deals drawn")
    ap.add_argument("--solver", default="dds", choices=["dds", "model"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", help="also write the results to this file")
    a = ap.parse_args()

    import torch
    from infer import Predictor
    torch.manual_seed(a.seed)
    P = Predictor(a.run, a.device)
    pos = P.position(a.hand, a.seat, a.dealer, a.vul, a.auction)
    t0 = time.time()
    q = Query(pos, a.hand, a.contracts.split(), a.n, a.solver, a.workers, a.seed, a.scoring)
    print(f"{a.hand} in {a.seat}, dealer {a.dealer}, vul {a.vul}, auction {a.auction or '(none)'}; model {os.path.basename(a.run)}; "
          f"{a.n} deals, {'solved double dummy' if a.solver == 'dds' else 'model trick odds'} ({time.time() - t0:.0f}s); scoring {a.scoring}")
    out = {"position": vars(a), "hypotheses": [], "asks": []}
    for h in a.hyps:
        r = q.hypothesis(h)
        print_hypothesis(r)
        out["hypotheses"].append(r)
    for s in a.ask:
        r = q.ask(s, a.hyps)
        print_ask(r)
        out["asks"].append(r)
    if not a.hyps and not a.ask:
        allr = np.ones(q.n, bool)
        for k in range(len(q.contracts)):
            v = q.summary(k, allr)
            print(f"  {q.contracts[k]['label']}: {v['ev']:+.0f}, makes {v['makes']:.0%}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
