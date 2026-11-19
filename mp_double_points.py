"""The matchpoint double, offline: at South's last turn of every auction of the Brill MP events that ended in an
undoubled East-West contract, what the auction model says of that contract and how usual South's own calls were;
then, against the real field and with the tricks really taken, what doubling there would have been worth under
different rules for when to double.

Stage 1, in a BEN container (src\\ben at /ben, this folder at /code, export BEN_ROOT=/ben AUCTION_MODEL_CODE=/code
PYTHONPATH=/var/lib/ben/torch-pkgs):

  python mp_double_points.py points --conf /var/lib/ben/brill/exp1/conf2/old-brillall-noadj.conf \
      --field /replay/field.jsonl --runs /data/runs/v1 /data/runs/v1-brill --out /data/tactical/mp_points.jsonl

Stage 2, anywhere with numpy:

  python mp_double_points.py report --field field.jsonl --points mp_points.jsonl [--out report.txt]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tactical import GAME, SEATS, Auction  # noqa: E402

W = {}
VUL = {"None": [False, False], "NS": [True, False], "EW": [False, True], "All": [True, True]}
MIN_TABLES = 8
# break-even chance of defeat by the share of the field in the same strain at this level or higher
# (mp_double_field.py, 4,690 tables of the Brill MP events, 2026-11-18)
BREAK_EVEN = ((0.10, 0.43), (0.30, 0.48), (0.60, 0.54), (0.90, 0.56), (1.01, 0.58))


def decisions(b):
    """Distinct auctions of a board that end in an undoubled East-West contract: (auction, South's last turn index)."""
    dealer = SEATS.index(b["dealer"])
    out = {}
    for t in b["tables"]:
        k = t["contract"].upper().replace("NT", "N")
        if not k[:1].isdigit() or t["declarer"] not in ("E", "W") or "X" in k[2:]:
            continue
        calls = t["auction"]
        last = max(i for i, c in enumerate(calls) if c[0].isdigit())
        j = next((i for i in range(last + 1, len(calls)) if (dealer + i) % 4 == 2), None)
        if j is not None:
            out["-".join(calls)] = j
    return out


# ------------------------------------------------------------------------------------------ stage 1

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
    W["preds"] = {os.path.basename(r.rstrip("/")): auction_model_deals._predictor(r)[0] for r in cfg["runs"]}


def do_board(b):
    try:
        return points_of(b), None
    except Exception:
        return [], f"{b['tournament_id']} {b['board']}\n{traceback.format_exc()}"


def points_of(b):
    from botbidder import BotBid
    from bidding import bidding
    from sample import Sample
    dealer = SEATS.index(b["dealer"])
    hand = b["deal"][2:].split()[2]
    vul_name = {"None": "None", "NS": "N_S", "EW": "E_W", "All": "Both"}[b["vul"]]
    soft_cache = {}

    def net(prefix):
        key = "-".join(prefix)
        if key not in soft_cache:
            auction = ["PAD_START"] * dealer + ["PASS" if c == "P" else c for c in prefix]
            bot = BotBid(VUL[b["vul"]], hand, W["models"], Sample.from_conf(W["conf"]), 2, dealer, W["dds"], False, False)
            soft_cache[key] = np.asarray(bot.next_bid_np(auction)[0]).reshape(-1)
        return soft_cache[key]

    out = []
    for text, j in decisions(b).items():
        calls = text.split("-")
        a = Auction(calls[:j], dealer, 2)
        _, level, strain, who = a.last
        decl = a.declarer(who, strain)
        probs = [float(net(calls[:i])[bidding.BID2ID["PASS" if calls[i] == "P" else calls[i]]])
                 for i in range(j) if (dealer + i) % 4 == 2]
        row = {"tournament_id": b["tournament_id"], "board": b["board"], "auction": text, "turn": j,
               "contract": f"{level}{strain}{SEATS[decl]}", "south_probs": [round(p, 5) for p in probs],
               "pass_prob": round(float(net(calls[:j])[bidding.BID2ID["PASS"]]), 5),
               "x_prob": round(float(net(calls[:j])[bidding.BID2ID["X"]]), 5)}
        for name, pred in W["preds"].items():
            pos = pred.position(hand, "S", b["dealer"], vul_name, calls[:j])
            rel = (decl - 2) % 4
            row[name] = [round(float(x), 4) for x in pos.dd[rel, "SHDCN".index(strain)]]
        out.append(row)
    return out


def stage_points(a):
    sys.path.insert(0, os.path.join(os.environ.get("BEN_ROOT", "/ben"), "scripts", "match"))
    import ab_match
    boards = [b for b in map(json.loads, open(a.field)) if b["scoring"] == "MP" and len(b["tables"]) >= MIN_TABLES]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    logs = a.out + ".logs"
    os.makedirs(logs, exist_ok=True)
    cfg = {"conf": ab_match.resolve(a.conf), "runs": a.runs, "logs": logs}
    t0, n, errors = time.time(), 0, 0
    with mp.get_context("spawn").Pool(a.workers, worker_init, (cfg,)) as pool, open(a.out, "w") as f:
        for rows, err in pool.imap_unordered(do_board, boards, chunksize=2):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            n += len(rows)
            if err:
                errors += 1
                open(os.path.join(logs, "errors.txt"), "a").write(err + "\n")
    print(f"done: {n} decision points on {len(boards)} MP boards, {errors} failed boards, {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------ stage 2

def matchpoints(s, field):
    f = np.asarray(field)
    return float(((f < s).sum() + 0.5 * (f == s).sum()) / len(f))


def stage_report(a):
    from scoring import STRAINS, contract_score
    pts = {(p["tournament_id"], p["board"], p["auction"]): p for p in map(json.loads, open(a.points))}
    names = [k for k in next(iter(pts.values())) if isinstance(next(iter(pts.values()))[k], list) and k != "south_probs"]
    rows = []
    n_boards = 0
    for line in open(a.field):
        b = json.loads(line)
        if b["scoring"] != "MP" or len(b["tables"]) < MIN_TABLES:
            continue
        n_boards += 1
        scores = [t["ns_score"] for t in b["tables"]]
        for i, t in enumerate(b["tables"]):
            p = pts.get((b["tournament_id"], b["board"], "-".join(t["auction"])))
            k = t["contract"].upper().replace("NT", "N")
            if p is None or not k[:1].isdigit() or t["declarer"] not in ("E", "W") or "X" in k[2:]:
                continue
            level, strain = int(k[0]), k[1]
            vul = VUL[b["vul"]][1]
            others = b["tables"][:i] + b["tables"][i + 1:]
            field = scores[:i] + scores[i + 1:]

            def same(o):
                ko = o["contract"].upper().replace("NT", "N")
                return ko[:1].isdigit() and o["declarer"] in ("E", "W") and ko[1] == strain and int(ko[0]) >= level

            sc = lambda dbl: -contract_score(level, STRAINS.index(strain), dbl, vul, t["tricks"])
            und = np.array([-contract_score(level, STRAINS.index(strain), 0, vul, x) for x in range(14)])
            dbl = np.array([-contract_score(level, STRAINS.index(strain), 1, vul, x) for x in range(14)])
            r = {"board": (b["tournament_id"], b["board"]), "date": b["date"], "failed": t["tricks"] < level + 6,
                 "gain": matchpoints(sc(1), field) - matchpoints(sc(0), field), "field_in": sum(same(o) for o in others) / len(others),
                 "path": float(np.prod(p["south_probs"])) if p["south_probs"] else 1.0,
                 "low": min(p["south_probs"]) if p["south_probs"] else 1.0, "x_prob": p["x_prob"], "game": level >= GAME[strain]}
            for name in names:
                d = np.array(p[name])
                r[name] = float(d[: level + 6].sum())                       # chance the contract fails
                r[name + "_points"] = float((d * (dbl - und)).sum())         # expected points gained by doubling
            rows.append(r)
    out = []
    say = out.append
    say(f"{len(rows)} tables with East-West in an undoubled contract on {n_boards} MP boards; North-South matchpoints 0-100, "
        f"real field, tricks really taken")
    say("gain = matchpoints added by doubling at the tables the rule doubles, summed and divided by ALL these tables; "
        "+- is a 95% interval over boards")

    def gain_of(sel_rows, mask):
        per = {}
        for r, m in zip(sel_rows, mask):
            per.setdefault(r["board"], []).append(100 * r["gain"] if m else 0.0)
        tot = np.array([sum(v) for v in per.values()])
        n = sum(len(v) for v in per.values())
        k = len(tot)
        se = np.sqrt(k / (k - 1) * ((tot - tot.mean()) ** 2).sum()) / n if k > 1 else float("nan")
        return tot.sum() / n, 1.96 * se

    def be(w):
        return next(p for top, p in BREAK_EVEN if w < top)

    for name in names:
        held = name.endswith("brill")
        R = [r for r in rows if (r["date"] > "2026-11-05" or not held)]
        say("")
        say(f"== model {name}" + (f" (only boards after 5 November: the model has seen the earlier deals; {len(R)} tables)" if held else f" ({len(R)} tables)"))
        p = np.array([r[name] for r in R])
        failed = np.array([r["failed"] for r in R])
        say("  chance of defeat, said -> true: " + ", ".join(
            f"{lo:.0%}-{hi:.0%}: {p[(p >= lo) & (p < hi)].mean():.0%} -> {failed[(p >= lo) & (p < hi)].mean():.0%} ({int(((p >= lo) & (p < hi)).sum())})"
            for lo, hi in ((0, .2), (.2, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.01)) if ((p >= lo) & (p < hi)).sum() >= 10))
        say(f"  {'rule for doubling':<58}{'doubles':>8}{'of them fail':>14}{'each':>8}{'gain':>8}{'+-':>6}")

        def row(label, mask):
            mask = np.asarray(mask)
            g, e = gain_of(R, mask)
            each = 100 * np.mean([r["gain"] for r, m in zip(R, mask) if m]) if mask.any() else float("nan")
            say(f"  {label:<58}{int(mask.sum()):>8}{(failed[mask].mean() if mask.any() else float('nan')):>13.0%}{each:>+8.1f}{g:>+8.2f}{e:>6.2f}")

        row("always", np.ones(len(R), bool))
        for th in (0.50, 0.58, 0.65, 0.75):
            row(f"chance of defeat at least {th:.0%}", p >= th)
        row("expected points gained by doubling at least 30 (today's rule)", np.array([r[name + "_points"] for r in R]) >= 30)
        w = np.array([r["field_in"] for r in R])
        row("field-aware, field known: defeat at least the break-even", p >= np.array([be(x) for x in w]))
        row("  the same, 5 points of margin", p >= np.array([be(x) for x in w]) + 0.05)
        path = np.array([r["path"] for r in R])
        low = np.array([r["low"] for r in R])
        say(f"  share of the field in the contract by how usual South's calls were (network's chance of South's least likely call): " + ", ".join(
            f"{lo:.0%}-{hi:.0%}: {w[(low >= lo) & (low < hi)].mean():.0%} ({int(((low >= lo) & (low < hi)).sum())})"
            for lo, hi in ((0, .02), (.02, .1), (.1, .3), (.3, .7), (.7, 1.01)) if ((low >= lo) & (low < hi)).sum() >= 10))
        guess = np.where(low < 0.02, 0.48, np.where(low < 0.10, 0.54, 0.58))
        row("field-aware, field guessed from how usual South's calls were", p >= guess)
        for lab, m in (("field in it under 30%", w < 0.30), ("field in it 30-90%", (w >= 0.30) & (w < 0.90)), ("field in it 90%+", w >= 0.90),
                       ("South's least likely call under 2%", low < 0.02), ("South's least likely call 2-10%", (low >= 0.02) & (low < 0.10))):
            for th in (0.45, 0.55):
                row(f"  {lab}: defeat at least {th:.0%}", m & (p >= th))
    text = "\n".join(out)
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    s1 = sub.add_parser("points")
    s1.add_argument("--conf", required=True)
    s1.add_argument("--field", required=True)
    s1.add_argument("--runs", nargs="+", required=True)
    s1.add_argument("--out", required=True)
    s1.add_argument("--workers", type=int, default=24)
    s2 = sub.add_parser("report")
    s2.add_argument("--field", required=True)
    s2.add_argument("--points", required=True)
    s2.add_argument("--out")
    a = ap.parse_args()
    stage_points(a) if a.stage == "points" else stage_report(a)
