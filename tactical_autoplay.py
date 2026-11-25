"""Does a gain measured double dummy survive real card play? For every deal on which two live replays
(brill_live.py) end in different contracts, Brill's /autoplay plays both contracts (its robots in all four seats),
and the two results are compared.

  elo/.venv/Scripts/python.exe auction-model/tactical_autoplay.py play --deals DEALS.jsonl --base BASE/replay.jsonl \
      --new NEW/replay.jsonl --out OUT.jsonl [--limit N] [--after-autoplay] [--give-way]
  elo/.venv/Scripts/python.exe auction-model/tactical_autoplay.py report --deals DEALS.jsonl --dd dd.json --out OUT.jsonl

play: one request at a time, about a minute each. Deals in a fixed random order, the baseline's contract and the
new one of a deal back to back, so an unfinished run is an unbiased sample of pairs. Can be started again: what
is in OUT is skipped; a contract that failed twice is left out. --after-autoplay waits until no other
brill_autoplay / tactical_autoplay process runs on this machine (the service is asked one board at a time).
--give-way makes the same check before every request, so a long run steps aside while the daily leaderboard
job (brill_autoplay.py) plays its boards.
"""
import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
import time
import urllib.error

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ben" / "scripts" / "match" / "brill"))
sys.path.insert(0, str(HERE))
import brill_autoplay  # noqa: E402
from score_brill_replay import contract_of, score_ns  # noqa: E402

GIVE_UP, COOL_DOWN = 2, 90


def others_running():
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match "
           "'brill_autoplay.py|tactical_autoplay.py' -and $_.ProcessId -ne %d -and $_.ParentProcessId -ne %d } | "
           "Measure-Object | Select-Object -ExpandProperty Count" % (os.getpid(), os.getpid()))
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout.strip()
    mine = subprocess.run(["powershell", "-NoProfile", "-Command",
                           "(Get-CimInstance Win32_Process -Filter \"ProcessId=%d\").ParentProcessId" % os.getpid()],
                          capture_output=True, text=True).stdout.strip()
    n = int(out or 0)
    # the venv launcher is this process's parent and carries the same command line
    if mine.isdigit():
        par = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Process -Filter \"ProcessId=%s\").CommandLine" % mine],
                             capture_output=True, text=True).stdout
        if "tactical_autoplay.py" in par:
            n -= 1
    return max(n, 0)


def pairs(a):
    deals = {(b["tournament_id"], b["board"]): b for b in map(json.loads, open(a.deals, encoding="utf-8"))}
    base = {(r["tournament_id"], r["board"]): r["walk"] for r in map(json.loads, open(a.base, encoding="utf-8")) if r.get("walk")}
    new = {(r["tournament_id"], r["board"]): r["walk"] for r in map(json.loads, open(a.new, encoding="utf-8")) if r.get("walk")}
    keys = sorted(k for k in base if k in new and contract_of(base[k], deals[k]["dealer"]) != contract_of(new[k], deals[k]["dealer"]))
    random.Random(20261119).shuffle(keys)
    return deals, [(k, base[k], new[k]) for k in keys]


def stage_play(a):
    deals, todo = pairs(a)
    if a.limit:
        todo = todo[: a.limit]
    out_path = pathlib.Path(a.out)
    done, failed = set(), {}
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            r = json.loads(line)
            key = (r["tournament_id"], r["board"], r["side"])
            if "error" in r:
                failed[key] = failed.get(key, 0) + 1
            else:
                done.add(key)
    jobs = [(k, side, walk) for k, wb, wn in todo for side, walk in (("base", wb), ("new", wn))
            if (k[0], k[1], side) not in done and failed.get((k[0], k[1], side), 0) < GIVE_UP]
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {len(todo)} deals with different contracts, {len(jobs)} contracts to play", flush=True)
    if a.after_autoplay:
        while others_running():
            time.sleep(120)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  no other autoplay job is running: starting", flush=True)
    status = brill_autoplay.get(f"{brill_autoplay.BASE}/ready")
    if status.get("status") != "ready":
        raise SystemExit(f"Brill service not ready: {status}")
    failures = slow = 0
    t_start = time.time()
    with out_path.open("a", encoding="utf-8") as out:
        for i, (k, side, walk) in enumerate(jobs):
            board = deals[k]
            row = {"tournament_id": k[0], "board": k[1], "side": side, "auction": walk}
            t0 = time.time()
            if contract_of(walk, board["dealer"]) is None:
                row.update({"contract": "PASS", "declarer": None, "tricks": 0, "ns_score": 0})
            else:
                if a.give_way and others_running():
                    print(f"{time.strftime('%H:%M:%S')}  another autoplay job is running: waiting", flush=True)
                    while others_running():
                        time.sleep(60)
                    print(f"{time.strftime('%H:%M:%S')}  going on", flush=True)
                    t0 = time.time()
                try:
                    r = brill_autoplay.autoplay(board, walk)
                    row.update({x: r[x] for x in ("contract", "declarer", "tricks", "ns_score", "service_ns_score")})
                    failures = slow = 0
                except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError, ValueError) as error:
                    row["error"] = f"{type(error).__name__}: {error}"[:300]
                    if isinstance(error, urllib.error.HTTPError) and error.code == 504:
                        slow += 1
                    else:
                        failures += 1
            row["seconds"] = round(time.time() - t0, 1)
            out.write(json.dumps(row) + "\n")
            out.flush()
            if (i + 1) % 10 == 0:
                print(f"{time.strftime('%H:%M:%S')}  {i + 1} of {len(jobs)} contracts, {(time.time() - t_start) / (i + 1):.0f} s each", flush=True)
            if failures >= 3 or slow >= 6:
                raise SystemExit(f"{max(failures, slow)} failures in a row, last: {row['error']}")
            if "error" in row:
                time.sleep(COOL_DOWN)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  done", flush=True)


IMP_STEPS = [20, 50, 90, 130, 170, 220, 270, 320, 370, 430, 500, 600, 750, 900, 1100, 1300, 1500, 1750, 2000, 2250,
             2500, 3000, 3500, 4000]


def imps(diff):
    d = np.asarray(diff, dtype=float)
    return np.sign(d) * np.searchsorted(IMP_STEPS, np.abs(d), side="right")


def ci(x):
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def stage_report(a):
    deals = {(b["tournament_id"], b["board"]): b for b in map(json.loads, open(a.deals, encoding="utf-8"))}
    dd = json.load(open(a.dd))
    played = {}
    for line in open(a.out, encoding="utf-8"):
        r = json.loads(line)
        if "error" not in r:
            played[(r["tournament_id"], r["board"], r["side"])] = r
    rows = []
    for k in sorted({(t, b) for t, b, _ in played}):
        if (k[0], k[1], "base") not in played or (k[0], k[1], "new") not in played:
            continue
        b = deals[k]
        rec = {}
        for side in ("base", "new"):
            p = played[(k[0], k[1], side)]
            c = contract_of(p["auction"], b["dealer"])
            if c is None:
                rec[side] = (None, 0, 0, 0, 0)
                continue
            dd_tricks = dd[f"{k[0]}|{k[1]}"]["tricks"][c[1] + c[3]]
            rec[side] = (c, p["ns_score"], score_ns(*c, dd_tricks, b["vul"]), p["tricks"], dd_tricks)
        cb, cn = rec["base"][0], rec["new"][0]
        kind = "double" if cb and cn and cb[:2] == cn[:2] and cb[3] == cn[3] and cn[2] and not cb[2] else "other"
        rows.append({"kind": kind, "real": rec["new"][1] - rec["base"][1], "dd": rec["new"][2] - rec["base"][2],
                     "tricks": [(rec[s][3], rec[s][4], rec[s][0][3] in "NS") for s in ("base", "new") if rec[s][0]]})
    n_all = len(deals)
    print(f"{len(rows)} deals with both contracts played by Brill's robots (new minus baseline, North-South view)")
    print(f"{'':<26}{'deals':>6}{'real play: IMPs each':>22}{'+-':>6}{'double dummy':>14}{'+-':>6}{'real, IMPs sum':>16}{'DD, IMPs sum':>14}")
    for lab, sel in (("all", rows), ("a double of the same contract", [r for r in rows if r["kind"] == "double"]),
                     ("a different contract", [r for r in rows if r["kind"] == "other"])):
        if len(sel) < 2:
            continue
        real, d = imps([r["real"] for r in sel]), imps([r["dd"] for r in sel])
        print(f"{lab:<26}{len(sel):>6}{real.mean():>+22.2f}{ci(real):>6.2f}{d.mean():>+14.2f}{ci(d):>6.2f}{real.sum():>+16.0f}{d.sum():>+14.0f}")
    real, d = imps([r["real"] for r in rows]), imps([r["dd"] for r in rows])
    print(f"better / same / worse in real play: {int((real > 0).sum())} / {int((real == 0).sum())} / {int((real < 0).sum())}; "
          f"double dummy: {int((d > 0).sum())} / {int((d == 0).sum())} / {int((d < 0).sum())}")
    t = [x for r in rows for x in r["tricks"]]
    for lab, sel in (("North-South declare", [x for x in t if x[2]]), ("East-West declare", [x for x in t if not x[2]])):
        if sel:
            diff = np.array([x[0] - x[1] for x in sel])
            print(f"declarer's tricks, real minus double dummy, {lab}: {diff.mean():+.2f} ({len(sel)} contracts; same {np.mean(diff == 0):.0%}, "
                  f"fewer {np.mean(diff < 0):.0%}, more {np.mean(diff > 0):.0%})")
    print(f"if the other deals behave the same, per deal over all {n_all} deals: scale the sums by the share of changed deals played")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("play")
    for k in ("deals", "base", "new", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--after-autoplay", action="store_true")
    p.add_argument("--give-way", action="store_true")
    r = sub.add_parser("report")
    for k in ("deals", "dd", "out"):
        r.add_argument("--" + k, required=True)
    a = ap.parse_args()
    stage_play(a) if a.stage == "play" else stage_report(a)
