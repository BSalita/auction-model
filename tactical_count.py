"""How often do the purpose rules of tactical.py name a call, and did BEN already have it as a candidate?

At every South turn of a replay (brill_replay.py or brill_live.py record: `south_calls` holds each auction South
faced and the call made) this runs BEN's candidate step (network above the threshold + EPBot's call, no search)
and the rules on the auction model's view of the same hand and auction.

Stage 1, in a BEN container with src\\ben at /ben, this folder at /code (export AUCTION_MODEL_CODE=/code):

  python tactical_count.py turns --conf /var/lib/ben/ab/conf/old.conf --deals /replay/field.jsonl \
      --replay /replay/runs/old.jsonl --run /data/runs/v1-brill --out /data/tactical/brill-old.jsonl --workers 48

Stage 2, anywhere with numpy (the report; --dd is a file of double-dummy tables as score_live_ab.py writes):

  python tactical_count.py report /data/tactical/brill-old.jsonl --dd /data/tactical/dd_brill.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import Counter, defaultdict

W = {}
SEATS = "NESW"
VUL = {"None": [False, False], "NS": [True, False], "EW": [False, True], "All": [True, True]}
RULES = ["toward", "interfere", "interfere2", "sacrifice", "sacrifice2", "double", "double2", "lead"]


def ben_call(c):
    return {"P": "PASS"}.get(c, c)


def worker_init(cfg):
    sys.path.insert(0, os.path.join(os.environ.get("BEN_ROOT", "/ben"), "scripts", "match"))
    import ab_match
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "2"
    os.chdir(ab_match.SRC)
    log = open(os.path.join(cfg["logs"], f"worker-{os.getpid()}.log"), "a")
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    from nn.models_tf2 import Models
    from ddsolver.ddsolver import DDSolver
    W["conf"] = ab_match.load_conf(cfg["conf"], None)
    W["models"] = Models.from_conf(W["conf"], ab_match.BEN_ROOT)
    ab_match.platform_fixups(W["models"])
    W["dds"] = DDSolver(max_threads=1)
    import auction_model_deals
    W["pred"] = auction_model_deals._predictor(cfg["run"])[0]        # also puts the model's folder on sys.path
    import tactical
    W["tactical"] = tactical


def do_board(task):
    try:
        return turns_of(*task), None
    except Exception:
        return [], f"{task[0]['tournament_id']} {task[0]['board']}\n{traceback.format_exc()}"


def turns_of(b, r):
    import numpy as np
    from botbidder import BotBid
    from bidding import bidding
    from sample import Sample
    dealer = SEATS.index(b["dealer"])
    hand = b["deal"][2:].split()[2]
    vuln = VUL[b["vul"]]
    walk = "-".join(r["walk"])
    out = []
    for key, made in r["south_calls"].items():
        calls = key.split("-") if key else []
        auction = ["PAD_START"] * dealer + [ben_call(c) for c in calls]
        np.random.seed(42)
        bot = BotBid(vuln, hand, W["models"], Sample.from_conf(W["conf"]), 2, dealer, W["dds"], False, False)
        bot.my_bid_no = bot.get_bid_number_for_player_to_bid(auction)
        cands, _ = bot.get_bid_candidates(auction)
        soft, _ = bot.next_bid_np(auction)
        soft = np.asarray(soft).reshape(-1)
        pos = W["pred"].position(hand, "S", b["dealer"], {"None": "None", "NS": "N_S", "EW": "E_W", "All": "Both"}[b["vul"]], calls)
        props, ctx = W["tactical"].propose(pos, hand, calls, dealer, 2, matchpoint=b.get("scoring") == "MP")
        for p in props:
            p["legal"] = bool(bidding.can_bid(p["bid"], auction))
            p["net"] = round(float(soft[bidding.BID2ID[p["bid"]]]), 5)
        out.append({"tournament_id": b["tournament_id"], "board": b["board"], "key": key, "hand": hand,
                    "dealer": b["dealer"], "vul": b["vul"], "made": made,
                    "on_walk": walk == key or walk.startswith(key + "-") or key == "",
                    "cands": [[("P" if c.bid == "PASS" else c.bid), round(float(c.insta_score), 4), getattr(c, "who", None) or "NN"]
                              for c in cands],
                    "props": props, "ctx": ctx})
    return out


def stage_turns(a):
    sys.path.insert(0, os.path.join(os.environ.get("BEN_ROOT", "/ben"), "scripts", "match"))
    import ab_match
    deals = {(b["tournament_id"], b["board"]): b for b in map(json.loads, open(a.deals))}
    tasks = []
    for line in open(a.replay):
        r = json.loads(line)
        b = deals[(r["tournament_id"], r["board"])]
        tasks.append(({k: b[k] for k in ("tournament_id", "board", "deal", "dealer", "vul", "scoring") if k in b}, r))
    if a.limit:
        tasks = tasks[: a.limit]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    logs = a.out + ".logs"
    os.makedirs(logs, exist_ok=True)
    cfg = {"conf": ab_match.resolve(a.conf), "run": a.run, "logs": logs}
    t0, n, errors = time.time(), 0, 0
    with mp.get_context("spawn").Pool(a.workers, worker_init, (cfg,)) as pool, open(a.out, "w") as f:
        for i, (turns, err) in enumerate(pool.imap_unordered(do_board, tasks, chunksize=4)):
            for t in turns:
                f.write(json.dumps(t) + "\n")
            n += len(turns)
            if err:
                errors += 1
                open(os.path.join(logs, "errors.txt"), "a").write(err + "\n")
            if (i + 1) % 500 == 0:
                print(f"{i + 1} of {len(tasks)} boards, {n} turns, {errors} failed boards, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {n} South turns on {len(tasks)} boards, {errors} failed boards, {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------- report

def dd_score(dd, contract, vul):
    """Double-dummy score for North-South of a contract like '4SxE' on the real deal."""
    from scoring import STRAINS, contract_score
    level, strain, decl = int(contract[0]), contract[1], contract[-1]
    dbl = contract.count("x")
    tricks = dd["tricks"][strain + decl]
    side_vul = VUL[vul][0 if decl in "NS" else 1]
    sc = contract_score(level, STRAINS.index(strain), dbl, side_vul, tricks)
    return sc if decl in "NS" else -sc, tricks >= level + 6


IMP_STEPS = [20, 50, 90, 130, 170, 220, 270, 320, 370, 430, 500, 600, 750, 900, 1100, 1300, 1500, 1750, 2000, 2250,
             2500, 3000, 3500, 4000]


def imps(diff):
    import numpy as np
    d = np.asarray(diff, dtype=float)
    return np.sign(d) * np.searchsorted(IMP_STEPS, np.abs(d), side="right")


def pct(x, n):
    return f"{100 * x / n:5.1f}%" if n else "    -"


def stage_report(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    turns = [json.loads(line) for line in open(a.file)]
    dd = json.loads(open(a.dd).read()) if a.dd else {}
    boards = {(t["tournament_id"], t["board"]) for t in turns}
    walks = {}
    if a.replay:
        for line in open(a.replay):
            r = json.loads(line)
            walks[(r["tournament_id"], r["board"])] = r["walk"]

    def ended(t):
        """BEN passed here in its own replay auction and nobody bid again: the board's result is this contract."""
        w = walks.get((t["tournament_id"], t["board"]))
        k = t["key"].split("-") if t["key"] else []
        return bool(w) and t["made"] == "P" and w[:len(k)] == k and len(w) > len(k) and all(c == "P" for c in w[len(k):])
    lines = []
    say = lines.append
    say(f"{len(turns)} South turns on {len(boards)} boards ({a.file})")
    n_c = Counter(len(t["cands"]) for t in turns)
    say("BEN's candidates a turn: " + ", ".join(f"{k}: {pct(v, len(turns)).strip()}" for k, v in sorted(n_c.items())))
    say("")
    say(f"{'rule / kind':<22}{'turns':>7}{'of all':>8}{'boards':>8}{'calls':>7}  {'candidate':>9}{'BEN made':>9}{'new':>7}"
        f"{'new turns':>10}{'net<1%':>8}")
    stat = defaultdict(lambda: {"turns": set(), "boards": set(), "calls": 0, "cand": 0, "made": 0, "new": 0,
                                "new_turns": set(), "tiny": 0})
    examples = defaultdict(list)
    truth = defaultdict(list)
    for i, t in enumerate(turns):
        cand = {c[0] for c in t["cands"]}
        d = dd.get(f"{t['tournament_id']}|{t['board']}")
        for p in t["props"]:
            if not p["legal"]:
                continue
            for name in (p["rule"], f"  {p['rule']}: {p['tag']}"):
                s = stat[name]
                s["turns"].add(i)
                s["boards"].add((t["tournament_id"], t["board"]))
                s["calls"] += 1
                s["cand"] += p["bid"] in cand
                s["made"] += p["bid"] == t["made"]
                if p["bid"] not in cand:
                    s["new"] += 1
                    s["new_turns"].add(i)
                    s["tiny"] += p["net"] < 0.01
            if p["bid"] not in cand:
                examples[f"{p['rule']}: {p['tag']}"].append((t, p))
            if d:
                name = f"{p['rule']}: {p['tag']}"
                new = p["bid"] not in cand
                if p["rule"] == "toward":
                    truth[name].append((new, p["make"], dd_score(d, p["target"], t["vul"])[1]))
                elif p["rule"] == "interfere":
                    ours, _ = dd_score(d, p["bid"] + "x" + p["decl"], t["vul"])
                    best = min(dd_score(d, f"{lvl}{s}{w}", t["vul"])[0] for s in "CDHSN" for lvl in range(1, 8) for w in "EW")
                    game = any(dd_score(d, f"{g}{s}{w}", t["vul"])[1] for s, g in zip("CDHSN", (5, 5, 4, 4, 3)) for w in "EW")
                    truth[name].append((new, t["ctx"]["p_game"], game, ours >= best, ours - best))
                elif p["rule"] == "interfere2":
                    level, strain = int(p["bid"][0]), p["bid"][1]
                    down = level + 6 - d["tricks"][strain + p["decl"]]
                    game = any(dd_score(d, f"{g}{s}{w}", t["vul"])[1] for s, g in zip("CDHSN", (5, 5, 4, 4, 3)) for w in "EW")
                    truth[name].append((new, t["ctx"]["p_game"], game, p["p_down3"], down >= 3, down <= 0))
                elif p["rule"] in ("sacrifice", "sacrifice2"):
                    ours, _ = dd_score(d, p["bid"] + "x" + p["decl"], t["vul"])
                    theirs, made = dd_score(d, p["their"], t["vul"])
                    truth[name].append((new, p["their_make"], made, ours > theirs, ours - theirs, ended(t)))
                elif p["rule"] in ("double", "double2"):
                    und, made = dd_score(d, p["their"], t["vul"])
                    dbl, _ = dd_score(d, p["their"][:2] + "x" + p["their"][-1], t["vul"])
                    truth[name].append((new, p["their_fail"], not made, dbl > und, dbl - und, ended(t)))
    for r in RULES:
        for name in [r] + sorted(k for k in stat if k.startswith(f"  {r}:")):
            s = stat.get(name)
            if not s:
                say(f"{name:<22}{0:>7}")
                continue
            say(f"{name:<22}{len(s['turns']):>7}{pct(len(s['turns']), len(turns)):>8}{len(s['boards']):>8}{s['calls']:>7}  "
                f"{pct(s['cand'], s['calls']):>9}{pct(s['made'], s['calls']):>9}{s['new']:>7}{len(s['new_turns']):>10}"
                f"{pct(s['tiny'], s['new']):>8}")
    any_new = {i for s in (stat[r] for r in RULES if r in stat) for i in s["new_turns"]}
    say("")
    say(f"Turns with at least one call BEN did not have as a candidate: {len(any_new)} ({pct(len(any_new), len(turns)).strip()}), "
        f"on {len({(turns[i]['tournament_id'], turns[i]['board']) for i in any_new})} boards")
    say("  columns: turns / boards where the rule names a legal call; calls named; share of them BEN already had as a")
    say("  candidate; share BEN made; calls and turns that are new to BEN; share of the new calls BEN's network rates under 1%")
    if truth:
        import numpy as np
        say("")
        say("On the real deal, double dummy. all: every named call; new: those BEN did not have as a candidate;")
        say("end: turns of BEN's own replay auction where BEN passed and that ended the auction (the difference is then")
        say("the change of the board's double-dummy score if the named call had ended the auction instead):")
        for name in sorted(truth):
            rows = truth[name]
            groups = [("all", rows), ("new", [x for x in rows if x[0]])]
            if len(rows[0]) == 6 and not name.startswith("interfere2"):
                groups.append(("end", [x for x in rows if x[5]]))
            for label, sel in groups:
                if not sel:
                    continue
                x = np.array([row[1:] for row in sel], dtype=float)
                if label == "end":
                    im = imps(x[:, 3])
                    say(f"  {name:<22}{label:>4} {len(sel):>6}: gains {x[:, 2].mean():.0%}, mean difference {x[:, 3].mean():+.0f} points, "
                        f"{im.mean():+.2f} IMPs a turn, {im.sum() / len(boards):+.3f} IMPs a board over all {len(boards)} boards")
                    continue
                if name.startswith("toward"):
                    say(f"  {name:<22}{label:>4} {len(sel):>6}: target contract makes, said {x[:, 0].mean():.0%}, true {x[:, 1].mean():.0%}")
                elif name.startswith("interfere2"):
                    say(f"  {name:<22}{label:>4} {len(sel):>6}: they have a game, said {x[:, 0].mean():.0%}, true {x[:, 1].mean():.0%}; "
                        f"our bid three or more down, said {x[:, 2].mean():.0%}, true {x[:, 3].mean():.0%}; our bid makes {x[:, 4].mean():.0%}")
                elif name.startswith("interfere"):
                    say(f"  {name:<22}{label:>4} {len(sel):>6}: they have a game, said {x[:, 0].mean():.0%}, true {x[:, 1].mean():.0%}; "
                        f"our bid doubled costs no more than their best contract {x[:, 2].mean():.0%}, mean difference {x[:, 3].mean():+.0f}")
                elif name.startswith("sacrifice"):
                    say(f"  {name:<22}{label:>4} {len(sel):>6}: their contract makes, said {x[:, 0].mean():.0%}, true {x[:, 1].mean():.0%}; "
                        f"our bid doubled beats defending {x[:, 2].mean():.0%}, mean difference {x[:, 3].mean():+.0f}")
                else:
                    say(f"  {name:<22}{label:>4} {len(sel):>6}: their contract fails, said {x[:, 0].mean():.0%}, true {x[:, 1].mean():.0%}; "
                        f"doubling gains {x[:, 2].mean():.0%}, mean difference {x[:, 3].mean():+.0f}")
    say("")
    say("Examples of calls new to BEN (hand S.H.D.C, dealer, vul, auction -> named call; what BEN did; its candidates):")
    import random
    rnd = random.Random(7)
    for name in sorted(examples):
        say(f" {name} ({len(examples[name])})")
        for t, p in rnd.sample(examples[name], min(a.examples, len(examples[name]))):
            facts = ", ".join(f"{k} {v}" for k, v in p.items() if k not in ("rule", "tag", "bid", "legal"))
            say(f"   {t['hand']:<17} {t['dealer']} {t['vul']:<4} {t['key'] or '(first call)':<28} -> {p['bid']:<3} BEN {t['made']:<3} "
                f"[{' '.join(c[0] for c in t['cands'])}]  {facts}; par {t['ctx']['par']}, their game {t['ctx']['p_game']:.0%}")
    text = "\n".join(lines)
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    t = sub.add_parser("turns")
    t.add_argument("--conf", required=True)
    t.add_argument("--deals", required=True)
    t.add_argument("--replay", required=True)
    t.add_argument("--run", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--workers", type=int, default=32)
    t.add_argument("--limit", type=int, default=0)
    r = sub.add_parser("report")
    r.add_argument("file")
    r.add_argument("--dd")
    r.add_argument("--replay", help="the replay the turns came from: adds the rows for turns that ended BEN's auction")
    r.add_argument("--out")
    r.add_argument("--examples", type=int, default=6)
    a = ap.parse_args()
    stage_turns(a) if a.stage == "turns" else stage_report(a)
