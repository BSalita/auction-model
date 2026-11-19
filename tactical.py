"""Candidate calls by purpose, from the auction model's view of one seat (infer.Position).

BEN's bidder only searches calls its network rates above a threshold (plus EPBot's call), so a call that no
system prescribes is never looked at. These rules name such calls from what the model expects of the deal:

  toward     a contract of ours is expected to score more than where the auction stands: the cheapest bid in
             its strain, and the contract itself
  interfere  the other side probably has a game and has not bid it yet: bids in our long suit (or a raise of
             partner's) up to the level where going down doubled still costs less than their best contract
  interfere2 the same moment, priced as the Brill robots really treat us (see P_DOUBLE): the one highest bid in
             the suit that, with the measured chance of a double, is expected to cost no more than their likeliest
             game and carries at most RISK of a penalty of 800 or more actually paid; never a one-level bid
  sacrifice  they have bid their contract (game or higher, or I am in the passout seat): a bid of ours that,
             doubled, is expected to cost less than defending ("sacrifice"), or costs little more while one
             level higher they are expected to fail ("push")
  sacrifice2 the same moment, priced with what the Brill robots really do after such a bid (robot_reaction.json:
             double, bid on, or pass it out, by game or partscore, tricks down and our side's points): a bid whose
             expected result over those three outcomes beats defending
  double     same moment: their contract doubled is expected to score more for us than undoubled
  double2    the double by the board's scoring: at matchpoints when the chance of defeat is at least MP_DOUBLE
             (break-even against the real Brill fields is 58%, mp_double_field.py; the model is a little too sure of
             defeats); at IMPs when the expected gain in IMPs over the trick distribution is at least IMP_DOUBLE
  lead       partner will probably be on lead: a cheap bid in a suit where I hold two of the top three
             honors, or a double of right-hand opponent's artificial bid of such a suit

Nothing here says the call is right; it only puts the call in front of the search. All scores are double dummy,
from the model's trick probabilities, for the viewpoint seat's side.
"""
import numpy as np

SEATS = "NESW"
RANK = "CDHSN"                     # bidding order of the strains
SUITS = "SHDC"                     # order in a hand string
GAME = {"C": 5, "D": 5, "H": 4, "S": 4, "N": 3}

MARGIN = 30                        # points a call must gain over the alternative to be proposed
P_GAME = 0.5                       # interfere: chance the other side makes some game
PUSH_COST = 100                    # push, lead: what the bid may cost, doubled, over defending
P_PUSH_NOW, P_PUSH_NEXT = 0.6, 0.45   # push: they make now / one level higher
ARTIFICIAL_LEN = 3.5               # lead double: RHO's expected length in the suit bid is under this
N_SETS = 128                       # sampled sets for that expected length, and for partner's expected points
MP_DOUBLE = 0.65                   # double2 at matchpoints: chance their contract fails
IMP_DOUBLE = 0.5                   # double2 at IMPs: expected IMPs gained by doubling
IMP_STEPS = [20, 50, 90, 130, 170, 220, 270, 320, 370, 430, 500, 600, 750, 900, 1100, 1300, 1500, 1750, 2000, 2250,
             2500, 3000, 3500, 4000]
RISK = 0.05                        # interfere2: chance of conceding 800 or more (doubled and that far down)
BIG_PENALTY = -800
# Share of our contracts the Brill robots doubled, by tricks down (0, 1, 2, 3, 4+), in auctions where both sides
# bid, by the points North-South held (up to 17, 18-22, 23+). Measured 2026-11-18 on 15,145 final contracts of
# the live replays (src/data/brill-replay/live/10k), smoothed where a cell had under 40 contracts.
P_DOUBLE = ((17, (0.23, 0.25, 0.32, 0.45, 0.45)), (22, (0.04, 0.05, 0.10, 0.21, 0.39)), (40, (0.01, 0.02, 0.02, 0.05, 0.10)))


def norm(c):
    c = c.strip().upper()
    return {"PASS": "P", "D": "X", "DBL": "X", "R": "XX", "RDBL": "XX"}.get(c, c)


class Auction:
    def __init__(self, calls, dealer, seat):
        self.calls = [norm(c) for c in calls]
        self.dealer, self.seat = dealer, seat
        assert (dealer + len(self.calls)) % 4 == seat, "not this seat's turn"
        self.bids = [(i, int(c[0]), c[1], (dealer + i) % 4) for i, c in enumerate(self.calls) if c[0].isdigit()]
        self.last = self.bids[-1] if self.bids else None
        self.dbl = 0
        if self.last:
            for c in self.calls[self.last[0] + 1:]:
                self.dbl = 1 if c == "X" else 2 if c == "XX" else self.dbl
        self.lho, self.partner, self.rho = (seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4

    def ours(self, who):
        return who % 2 == self.seat % 2

    def cheapest(self, strain):
        """Lowest level at which `strain` can be bid now, or None."""
        if not self.last:
            return 1
        lvl = self.last[1] if RANK.index(strain) > RANK.index(self.last[2]) else self.last[1] + 1
        return lvl if lvl <= 7 else None

    def declarer(self, side_seat, strain):
        """Who of that side named the strain first, or None."""
        for _, _, s, who in self.bids:
            if s == strain and who % 2 == side_seat % 2:
                return who
        return None

    def strains_of(self, who):
        return {s for _, _, s, w in self.bids if w == who}

    @property
    def theirs_holds(self):
        return self.last is not None and not self.ours(self.last[3])

    @property
    def final_theirs(self):
        """They hold the contract and it looks final: game or higher, or two passes to me."""
        if not self.theirs_holds:
            return False
        return self.last[1] >= GAME[self.last[2]] or self.calls[-2:] == ["P", "P"]

    @property
    def they_bid_game(self):
        return any(lvl >= GAME[s] for _, lvl, s, who in self.bids if not self.ours(who))


class View:
    """Expected scores for my side, read from a Position."""

    def __init__(self, pos, auction):
        self.pos, self.a = pos, auction

    def ev(self, level, strain, declarer, dbl=0):
        """(expected score for my side, chance the result is the good one for my side: ours makes / theirs fails)."""
        sc, good = self.pos.contract(f"{level}{strain}{('', 'x', 'xx')[dbl]}", SEATS[declarer])
        return float(sc), float(good)

    def our_declarer(self, strain):
        d = self.a.declarer(self.a.seat, strain)
        return self.a.seat if d is None else d

    def our(self, level, strain, dbl=0):
        return self.ev(level, strain, self.our_declarer(strain), dbl)

    def their(self, level, strain, dbl=0):
        """Their contract with the declarer who named the strain first; unnamed: the better of the two for them."""
        d = self.a.declarer(self.a.lho, strain)
        if d is not None:
            return self.ev(level, strain, d, dbl) + (d,)
        return min((self.ev(level, strain, d, dbl) + (d,) for d in (self.a.lho, self.a.rho)), key=lambda t: t[0])

    def now(self):
        """Expected score if the auction ended here; None when nobody has bid."""
        if self.a.last is None:
            return 0.0
        _, lvl, s, who = self.a.last
        if self.a.ours(who):
            return self.our(lvl, s, self.a.dbl)[0]
        return self.their(lvl, s, self.a.dbl)[0]

    def their_game(self):
        """(chance they make their likeliest game, strain, declarer) and the chance of their likeliest slam."""
        best = max(((1 - g, s, d) for s in RANK for _, g, d in [self.their(GAME[s], s)]), key=lambda t: t[0])
        slam = max(1 - self.their(6, s)[1] for s in RANK)
        return best, slam

    def their_best(self):
        """Their contract that is expected to cost us most: (score for us, level, strain, declarer)."""
        return min(((sc, lvl, s, d) for s in RANK for lvl in range(1, 8) for sc, _, d in [self.their(lvl, s)]),
                   key=lambda t: t[0])


def _points(hand):
    return sum(4 * hand.count("A") + 3 * hand.count("K") + 2 * hand.count("Q") + hand.count("J") for hand in [hand.upper()])


def priced(pos, level, strain, declarer, p_double):
    """Our contract with the measured chance of being doubled at each number of tricks down:
    (expected score for my side, chance of a doubled penalty of 800 or more, chance of three or more down)."""
    import sys
    mod = sys.modules[type(pos).__module__]                 # the auction model's infer module: its score table
    si = mod.STRAINS.index(strain)
    rel = (declarer - pos.seat) % 4
    p = pos.dd[rel, si]
    vul = int(pos.side_vul(rel))
    und, dbl = mod.SCORE[level - 1, si, 0, vul], mod.SCORE[level - 1, si, 1, vul]
    down = np.clip(level + 6 - np.arange(14), 0, 4)
    pd = np.asarray(p_double)[down]
    return (float((p * (pd * dbl + (1 - pd) * und)).sum()), float((p * pd * (dbl <= BIG_PENALTY)).sum()),
            float(p[: max(0, level + 4)].sum()))


_REACTION = None


def reaction(game, ns_pts):
    """[(chance of a double, chance they bid on) for 0..4+ tricks down] after our bid over their game / partscore."""
    global _REACTION
    if _REACTION is None:
        import json
        import os
        _REACTION = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_reaction.json")))
    rows = _REACTION["game" if game else "part"]
    return np.array(rows[next(k for k in ("17", "22", "40") if ns_pts <= int(k))])


def priced3(pos, level, strain, declarer, react, on_score):
    """Our bid over their contract with the robots' measured reaction: (expected score for my side, chance of a doubled
    penalty of 800 or more, chance left to play doubled, chance they bid on). on_score: what we expect when they
    bid on (None: they cannot; that share is then left to play undoubled)."""
    import sys
    mod = sys.modules[type(pos).__module__]
    si = mod.STRAINS.index(strain)
    rel = (declarer - pos.seat) % 4
    p = pos.dd[rel, si]
    vul = int(pos.side_vul(rel))
    und, dbl = mod.SCORE[level - 1, si, 0, vul], mod.SCORE[level - 1, si, 1, vul]
    down = np.clip(level + 6 - np.arange(14), 0, 4)
    pd, po = react[down, 0], react[down, 1]
    if on_score is None:
        po, on_score = np.zeros(14), 0.0
    ev = (p * (pd * dbl + po * on_score + (1 - pd - po) * und)).sum()
    return float(ev), float((p * pd * (dbl <= BIG_PENALTY)).sum()), float((p * pd).sum()), float((p * po).sum())


def _suits(hand):
    return dict(zip(SUITS, hand.upper().split(".")))


def imps(diff):
    d = np.asarray(diff, dtype=float)
    return np.sign(d) * np.searchsorted(IMP_STEPS, np.abs(d), side="right")


def propose(pos, hand, calls, dealer, seat, matchpoint=None):
    """-> (proposals, context). A proposal: {'rule', 'tag', 'bid', ...numbers}. Bids as '2S', 'X'.
    matchpoint: the board's scoring (True MP, False IMPs, None unknown = IMPs) for the rules that depend on it."""
    a = Auction(calls, dealer, seat)
    v = View(pos, a)
    mine = _suits(hand)
    now = v.now()
    par = float(pos.par_summary()[0])
    (p_game, g_strain, g_decl), p_slam = v.their_game()
    worst, w_lvl, w_strain, w_decl = v.their_best()
    ctx = {"par": round(par), "now": round(now), "p_game": round(p_game, 3), "p_slam": round(p_slam, 3),
           "their_best": f"{w_lvl}{w_strain}{SEATS[w_decl]}", "their_best_score": round(worst),
           "final_theirs": a.final_theirs}
    out = []
    opp_strains = a.strains_of(a.lho) | a.strains_of(a.rho)
    partner_strains = a.strains_of(a.partner)
    # suits we might play in: mine of 5+, partner's with 3+ support
    own = {s: len(mine[s]) for s in SUITS if s not in opp_strains
           and (len(mine[s]) >= 5 or (s in partner_strains and len(mine[s]) >= 3))}

    # ---- toward a contract of ours --------------------------------------------------------------------
    best = None
    for s in RANK:
        lo = a.cheapest(s)
        for lvl in range(lo, 8) if lo else ():
            sc, make = v.our(lvl, s)
            if best is None or sc > best[0]:
                best = (sc, make, lvl, s, lo)
    if best and best[0] >= now + MARGIN and best[1] >= 0.5:
        sc, make, lvl, s, lo = best
        d = a.declarer(seat, s)
        fact = {"target": f"{lvl}{s}{SEATS[seat if d is None else d]}", "score": round(sc), "make": round(make, 3)}
        out.append({"rule": "toward", "tag": "cheapest", "bid": f"{lo}{s}", **fact})
        if lvl != lo:
            out.append({"rule": "toward", "tag": "contract", "bid": f"{lvl}{s}", **fact})

    # ---- interfere: they probably have a game they have not bid ----------------------------------------
    if p_game >= P_GAME and not a.they_bid_game and not a.final_theirs:
        for s, n in own.items():
            lo = a.cheapest(s)
            if lo is None or (n == 5 and s not in partner_strains and lo > 2):
                continue
            top = None
            for lvl in range(lo, min(lo + 3, 8)):
                cost = v.our(lvl, s, 1)[0]
                if cost < worst:
                    break
                top = (lvl, cost)
            for lvl in range(lo, top[0] + 1) if top else ():
                out.append({"rule": "interfere", "tag": "raise" if s in partner_strains else "own suit",
                            "bid": f"{lvl}{s}", "decl": SEATS[v.our_declarer(s)],
                            "cost": round(v.our(lvl, s, 1)[0]), "length": n,
                            "their_game": f"{GAME[g_strain]}{g_strain}{SEATS[g_decl]}"})

    # ---- interfere2: the same, priced with the robots' real doubling, one bid a suit ---------------------
    if p_game >= P_GAME and not a.they_bid_game and not a.final_theirs and own:
        hon = pos.samples(N_SETS)["hon"]
        partner_pts = float(((hon == 2) * np.tile([4, 3, 2, 1], 4)).sum(1).mean())
        ns_pts = _points(hand) + partner_pts
        p_double = next(row for top, row in P_DOUBLE if ns_pts <= top)
        ref = v.their(GAME[g_strain], g_strain)[0]              # their likeliest game, as they would play it
        for s, n in own.items():
            lo = a.cheapest(s)
            if lo is None:
                continue
            raise_ = s in partner_strains
            top = None
            for lvl in range(max(lo, 2), 8):                    # a one-level bid takes no room
                if a.last is None and (lvl, s) == (2, "C"):     # the strong artificial opening
                    continue
                if n == 5 and not raise_ and lvl != lo:         # a five-card suit only without a jump
                    break
                ev, risk, down3 = priced(pos, lvl, s, v.our_declarer(s), p_double)
                if ev < ref or risk > RISK:
                    break
                top = (lvl, ev, risk, down3)
            if top:
                out.append({"rule": "interfere2", "tag": "raise" if raise_ else "own suit", "bid": f"{top[0]}{s}",
                            "decl": SEATS[v.our_declarer(s)], "priced": round(top[1]), "their_game_score": round(ref),
                            "risk": round(top[2], 3), "p_down3": round(top[3], 3), "ns_points": round(ns_pts, 1),
                            "length": n, "their_game": f"{GAME[g_strain]}{g_strain}{SEATS[g_decl]}"})

    # ---- they have bid their contract: sacrifice, push, double -----------------------------------------
    if a.final_theirs:
        _, c_lvl, c_strain, _ = a.last
        c_sc, c_fail, c_decl = v.their(c_lvl, c_strain, a.dbl)
        contract = f"{c_lvl}{c_strain}{('', 'x', 'xx')[a.dbl]}{SEATS[c_decl]}"
        for s in sorted(set(own) | {x for x in a.strains_of(seat) | partner_strains if x != "N"}):
            lo = a.cheapest(s)
            if lo is None or s in opp_strains:
                continue
            cost = v.our(lo, s, 1)[0]
            tags = []
            if cost >= c_sc + MARGIN:
                tags.append("sacrifice")
            # their next bid in their strain over ours
            n_lvl = lo if RANK.index(c_strain) > RANK.index(s) else lo + 1
            if n_lvl <= 7 and cost >= c_sc - PUSH_COST and 1 - c_fail >= P_PUSH_NOW \
                    and 1 - v.their(n_lvl, c_strain)[1] <= P_PUSH_NEXT:
                tags.append("push")
            for tag in tags:
                out.append({"rule": "sacrifice", "tag": tag, "bid": f"{lo}{s}", "decl": SEATS[v.our_declarer(s)],
                            "cost": round(cost),
                            "their": contract, "defend": round(c_sc), "their_make": round(1 - c_fail, 3)})
        if a.dbl == 0:
            x_sc = v.their(c_lvl, c_strain, 1)[0]
            if x_sc >= c_sc + MARGIN:
                out.append({"rule": "double", "tag": "penalty", "bid": "X", "their": contract,
                            "defend": round(c_sc), "doubled": round(x_sc), "their_fail": round(c_fail, 3)})

    # ---- double2: the double by the board's scoring ----------------------------------------------------
    if a.final_theirs and a.dbl == 0:
        _, c_lvl, c_strain, _ = a.last
        c_sc, c_fail, c_decl = v.their(c_lvl, c_strain)
        import sys
        mod = sys.modules[type(pos).__module__]
        si, rel = mod.STRAINS.index(c_strain), (c_decl - seat) % 4
        vul = int(pos.side_vul(rel))
        gain = float((pos.dd[rel, si] * imps(mod.SCORE[c_lvl - 1, si, 0, vul] - mod.SCORE[c_lvl - 1, si, 1, vul])).sum())
        if (c_fail >= MP_DOUBLE) if matchpoint else (gain >= IMP_DOUBLE):
            out.append({"rule": "double2", "tag": "matchpoints" if matchpoint else "IMPs", "bid": "X",
                        "their": f"{c_lvl}{c_strain}{SEATS[c_decl]}", "defend": round(c_sc),
                        "doubled": round(v.their(c_lvl, c_strain, 1)[0]), "their_fail": round(c_fail, 3),
                        "imp_gain": round(gain, 2)})

    # ---- sacrifice2: the same moment, priced with the robots' measured reaction -------------------------
    if a.final_theirs:
        _, c_lvl, c_strain, _ = a.last
        c_sc, c_fail, c_decl = v.their(c_lvl, c_strain, a.dbl)
        suits = sorted(set(own) | {x for x in a.strains_of(seat) | partner_strains if x != "N"})
        suits = [s for s in suits if s not in opp_strains and a.cheapest(s)]
        if suits:
            hon = pos.samples(N_SETS)["hon"]
            ns_pts = _points(hand) + float(((hon == 2) * np.tile([4, 3, 2, 1], 4)).sum(1).mean())
            game = c_lvl >= GAME[c_strain]
            react = reaction(game, ns_pts)
            for s in suits:
                lo = a.cheapest(s)
                n_lvl = lo if RANK.index(c_strain) > RANK.index(s) else lo + 1
                on = v.their(n_lvl, c_strain)[0] if n_lvl <= 7 else None
                ev, risk, p_x, p_on = priced3(pos, lo, s, v.our_declarer(s), react, on)
                if ev >= c_sc + MARGIN and risk <= RISK:
                    out.append({"rule": "sacrifice2", "tag": "over game" if game else "over partscore", "bid": f"{lo}{s}",
                                "decl": SEATS[v.our_declarer(s)], "priced": round(ev), "cost": round(v.our(lo, s, 1)[0]),
                                "their": f"{c_lvl}{c_strain}{('', 'x', 'xx')[a.dbl]}{SEATS[c_decl]}", "defend": round(c_sc),
                                "their_make": round(1 - c_fail, 3), "p_doubled": round(p_x, 3), "p_bid_on": round(p_on, 3),
                                "ns_points": round(ns_pts, 1)})

    # ---- lead direction: partner will probably lead ----------------------------------------------------
    if a.theirs_holds and par <= 0:
        t = a.last[2]
        strong = [s for s in SUITS if sum(h in mine[s] for h in "AKQ") >= 2 and len(mine[s]) >= 4]
        if a.declarer(a.lho, t) == a.lho:
            for s in strong:
                lo = a.cheapest(s)
                if s == t or s in opp_strains or lo is None or lo > 3:
                    continue
                cost = v.our(lo, s, 1)[0]
                if cost >= now - PUSH_COST:
                    out.append({"rule": "lead", "tag": "bid", "bid": f"{lo}{s}", "cost": round(cost),
                                "defend": round(now), "holding": mine[s]})
        # RHO has just bid a suit that looks artificial, LHO has bid: double it for the lead
        if a.last[0] == len(a.calls) - 1 and a.last[3] == a.rho and a.dbl == 0 and a.last[2] in strong \
                and a.strains_of(a.lho):
            rho_len = pos.marginals(N_SETS)[1]["RHO"][t]
            if rho_len < ARTIFICIAL_LEN:
                out.append({"rule": "lead", "tag": "double", "bid": "X", "holding": mine[t],
                            "rho_length": round(rho_len, 2), "defend": round(now)})
    return out, ctx
