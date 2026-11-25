"""Tricks after the anticipated opening lead, for the explorer page.

The model's trick forecasts are double dummy: the defence finds its best lead. This module answers what changes
when the leader instead leads as Morty's lead network would. The page draws full deals from the model
(`sample_deals`), and a helper process running BEN's own code (`Helper`, started with `--serve`) takes each deal's
leader hand through BEN's lead network, solves every opening lead double dummy, and returns for each contract
the distribution of declarer's tricks after the defence's best lead (`dd`) and after the anticipated lead
(`lead`: every lead the leader can make, by the network's share). Both are over the same deals, so their
difference is the effect of the lead alone.

The helper is a process of its own because BEN's code needs TensorFlow and module names (scoring, ...) that
this folder has too. It runs where BEN is installed (BEN_ROOT, default /opt/ben: the ben image); elsewhere
`Helper()` raises Unavailable and the page says so. It uses the lead and information networks of BEN's REST API
settings, with the lines of /var/lib/ben/api_override.conf replaced as BEN's entrypoint does.
"""
import json
import os
import subprocess
import sys
import threading

import numpy as np

SEATS = "NESW"
RANKS = "AKQJT98765432"
BEN_ROOT = os.environ.get("BEN_ROOT", "/opt/ben")
OVERRIDE = os.environ.get("BEN_API_OVERRIDE", "/var/lib/ben/api_override.conf")


class Unavailable(Exception):
    pass


# ---- the page's side ------------------------------------------------------------------------------------
def sample_deals(pos, hand, seat, n, rng):
    """n full deals drawn from the model's position `pos` (the view of `seat` holding `hand`), as PBN strings
    'N:... ... ... ...'. The model gives the suit lengths and the A, K, Q, J of the hidden hands; the cards below
    the jack are dealt at random within the lengths."""
    s = pos.samples(n)
    lens, hon = np.asarray(s["lens"]), np.asarray(s["hon"])
    mine = hand.upper().split(".")
    me = SEATS.index(seat)
    deals = []
    for i in range(n):
        hidden = {r: [[] for _ in range(4)] for r in (1, 2, 3)}          # seat after mine by r -> suit -> ranks
        for su in range(4):
            for j, rank in enumerate("AKQJ"):
                r = int(hon[i, su * 4 + j])
                if r:
                    hidden[r][su].append(rank)
            loose = [c for c in "T98765432" if c not in mine[su]]
            loose = [loose[k] for k in rng.permutation(len(loose))]
            a = 0
            for r in (1, 2, 3):
                room = int(lens[i, su, r - 1]) - len(hidden[r][su])
                hidden[r][su] += loose[a:a + room]
                a += room
        text = {0: hand.upper()}
        for r in (1, 2, 3):
            text[r] = ".".join("".join(sorted(x, key=RANKS.index)) for x in hidden[r])
        deals.append("N:" + " ".join(text[(k - me) % 4] for k in range(4)))
    return deals


class Helper:
    """The helper process. ask(deals, dealer, vul, cases) -> one answer a case; see `forecast`."""

    def __init__(self, log=None):
        if not os.path.isfile(os.path.join(BEN_ROOT, "src", "botbidder.py")):
            raise Unavailable(f"BEN is not installed here ({BEN_ROOT})")
        self.lock = threading.Lock()
        self.log = log or os.path.join(os.environ.get("TMPDIR", "/tmp"), "lead_forecast.log")
        self.proc = None
        self.start()

    def start(self):
        # BEN runs on the image's own packages, not on the page's
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        self.proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--serve"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=open(self.log, "a"), env=env, text=True)
        ready = self.proc.stdout.readline()
        if not ready:
            raise Unavailable(f"the lead helper did not start (see {self.log})")
        self.info = json.loads(ready)

    def ask(self, deals, dealer, vul, cases):
        with self.lock:
            if self.proc.poll() is not None:
                self.start()
            self.proc.stdin.write(json.dumps({"deals": deals, "dealer": dealer, "vul": vul, "cases": cases}) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        if not line:
            raise Unavailable(f"the lead helper stopped (see {self.log})")
        answer = json.loads(line)
        if "error" in answer:
            raise RuntimeError(answer["error"])
        return answer["cases"]


# ---- BEN's side -----------------------------------------------------------------------------------------
def effective_conf(src):
    """BEN's REST API settings as the robot of this machine reads them: config/default_api.conf with the [api]
    lines of the override file replaced (as ops/ben_entrypoint.sh does; a key that is not there or a network file
    that is missing is left out instead of stopping)."""
    import re
    from configparser import ConfigParser
    path = os.path.join(src, "config", "default_api.conf")
    if not os.path.isfile(OVERRIDE):
        return path, []
    override = ConfigParser(interpolation=None)
    override.optionxform = str
    override.read(OVERRIDE)
    if not override.has_section("api"):
        return path, []
    keys = {k: v for k, v in override["api"].items() if not (v.endswith(".keras") and not os.path.isfile(v))}
    used, out = [], []
    with open(path) as f:
        for line in f:
            key = re.match(r"(\w+)\s*=", line)
            if key and key.group(1) in keys:
                line = f"{key.group(1)} = {keys[key.group(1)]}\n"
                used.append(key.group(1))
            out.append(line)
    made = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"lead_forecast_api_{os.getpid()}.conf")
    with open(made, "w") as f:
        f.writelines(out)
    return made, used


def serve():
    import traceback
    from concurrent.futures import ThreadPoolExecutor
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)                                   # whatever BEN prints goes to the log, not into the answers
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    src = os.path.join(BEN_ROOT, "src")
    os.chdir(src)
    sys.path.insert(0, src)
    import binary
    import conf as ben_conf
    from bidding import bidding
    from ddsolver import ddsolver as solver
    from nn.models_tf2 import Models
    from util import follow_suit

    conf_path, replaced = effective_conf(src)
    models = Models.from_conf(ben_conf.load(conf_path), BEN_ROOT)
    n_play, n_bid = models.n_cards_play, models.n_cards_bidding
    per_suit = n_play // 4
    parse_play, parse_bid = binary.parse_hand_f(n_play), binary.parse_hand_f(n_bid)
    pool = ThreadPoolExecutor(max_workers=min(12, os.cpu_count() or 4))
    rank_bits = [0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100, 0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004]

    def solve_leads(strain, leader, deals):
        """A deal: {card code: the defence's tricks after that lead} for every card the leader holds."""
        def one(pbn):
            fut = solver.dds3.solve_board_pbn(pbn, trump=(strain - 1) % 5, first=leader, current_trick_suit=(0, 0, 0),
                                              current_trick_rank=(0, 0, 0), target=-1, solutions=3, mode=1,
                                              context=solver._thread_context())
            cards = {}
            for i in range(fut["cards"]):
                suit = fut["suit"][i]
                cards[suit * 13 + 14 - fut["rank"][i]] = fut["score"][i]
                for k, bit in enumerate(rank_bits):
                    if bit & fut["equals"][i]:
                        cards[suit * 13 + k] = fut["score"][i]
            return cards
        return list(pool.map(one, deals))

    def lead_inputs(auction, contract, leader, leader_hands, vuln, dealer):
        """The lead network's inputs for every leader hand: what binary.get_auction_binary_for_lead builds a hand,
        with the information network asked once for all hands. The first hand is also built by BEN's own function;
        if the two differ (BEN's function changed), every hand is built by BEN's function."""
        n = len(leader_hands)
        held = np.concatenate([parse_play(h) for h in leader_hands])
        bid_hands = np.concatenate([parse_bid(h) for h in leader_hands])
        X, B = np.zeros((n, 10 + n_play)), np.zeros((n, 15))
        X[:, 0] = int(contract[0])
        X[:, 1 + bidding.get_strain_i(contract)] = 1
        X[:, 6], X[:, 7] = int("X" in contract), int("XX" in contract)
        X[:, 8], X[:, 9] = vuln[leader % 2], vuln[(leader + 1) % 2]
        X[:, 10:] = held
        steps = binary.calculate_step_bidding_info(auction)
        A = binary.get_auction_binary_sampling(steps, auction, leader, bid_hands, vuln, models, n_bid)
        p_hcp, p_shp = models.binfo_model.pred_fun(A)
        B[:, :3] = np.asarray(p_hcp).reshape((n, steps, 3))[:, -1, :]
        B[:, 3:] = np.asarray(p_shp).reshape((n, steps, 12))[:, -1, :]
        x0, b0 = binary.get_auction_binary_for_lead(auction, bid_hands[:1], held[:1], vuln, dealer, models)
        if not (np.allclose(X[0], np.asarray(x0, dtype=np.float64).reshape(-1), atol=1e-3)
                and np.allclose(B[0], np.asarray(b0, dtype=np.float64).reshape(-1), atol=2e-2)):
            sys.stderr.write("lead_inputs: batch differs from BEN's function, building hand by hand\n")
            for i in range(n):
                x, b = binary.get_auction_binary_for_lead(auction, bid_hands[i:i + 1], held[i:i + 1], vuln, dealer, models)
                X[i], B[i] = np.asarray(x).reshape(-1), np.asarray(b).reshape(-1)
        return X, B, held

    def forecast(req):
        deals = req["deals"]
        hands = [d[2:].split() for d in deals]
        dealer = SEATS.index(req["dealer"])
        n = len(deals)
        answers = []
        for case in req["cases"]:
            auction = ["PAD_START"] * dealer + case["auction"]
            contract = bidding.get_contract(auction)
            if contract is None:
                answers.append(None)
                continue
            strain = "NSHDC".index(contract[1])
            leader = (SEATS.index(contract[-1]) + 1) % 4
            X, B, held = lead_inputs(auction, contract, leader, [h[leader] for h in hands], req["vul"], dealer)
            net = models.lead_nt_model if strain == 0 else models.lead_suit_model
            shares = follow_suit(net.pred_fun(X, B)[:, :n_play], held, np.zeros((n, 4)), n_play)
            dd, lead, by_suit = np.zeros(14), np.zeros(14), np.zeros(4)
            for i, cards in enumerate(solve_leads(strain, leader, deals)):
                dd[13 - max(cards.values())] += 1
                dist = np.zeros(14)
                for c in np.nonzero(shares[i] > 0)[0]:
                    suit, rank = divmod(int(c), per_suit)
                    if rank < per_suit - 1:
                        defence = cards.get(suit * 13 + rank)
                    else:                             # a small card: the one of them that is best for the defence
                        small = [v for card, v in cards.items() if card // 13 == suit and card % 13 >= per_suit - 1]
                        defence = max(small) if small else None
                    if defence is not None:
                        dist[13 - defence] += shares[i, c]
                        by_suit[suit] += shares[i, c] / n
                lead += dist / dist.sum() if dist.sum() > 0 else np.eye(14)[13 - max(cards.values())]
            answers.append({"contract": contract, "dd": (dd / n).tolist(), "lead": (lead / n).tolist(),
                            "suits": by_suit.tolist()})
        return {"cases": answers}

    out.write(json.dumps({"ready": True, "conf": conf_path, "replaced": replaced, "name": models.name}) + "\n")
    out.flush()
    for line in sys.stdin:
        try:
            answer = forecast(json.loads(line))
        except Exception:
            answer = {"error": traceback.format_exc()}
        out.write(json.dumps(answer) + "\n")
        out.flush()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        serve()
