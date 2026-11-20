"""North-dealer boards from the raw BBO downloads (.lin files), which the BBO table lost.

bbo_mldf_augmented.parquet has no board dealt by North (boards 1, 5, 9, 13 are absent from the first table built
from the downloads onwards), while the downloads have them. A model trained on that table never sees the best hand
opposite the dealer, and so reads "my partner dealt and passed" as "an opponent is stronger than I am".

  python extract_bbo_lin.py scan  --out DIR [--share 1.0] [--workers 64] [--limit-dirs N]
        reads the .lin files under data\\<player>\\ of the BBO folder (paths.py) and writes DIR\\lin_north.parquet:
        one row per distinct deal + vulnerability + auction dealt by North (PBN, Vul, auction, n tables)
  python extract_bbo_lin.py dd    --out DIR [--workers 100]
        double-dummy tricks and par of every distinct deal -> DIR\\lin_north_dd.parquet (resumable)
  python extract_bbo_lin.py build --out DIR
        -> DIR\\bbo_north_train.npz, bbo_north_test.npz (same arrays and held-out rule as extract_bbo.py) and
        bbo4_train.npz, bbo4_test.npz = the old files plus these rows

--share: the share of player folders read (a deal is played at about 26 tables, so a part of the folders already
finds most deals, with fewer of their auctions).
"""
import argparse
import os
import random
import sys
import time

import numpy as np

import paths

RAW = os.path.join(paths.BBO, "data")
SEATS = "NESW"
LIN_VUL = {"o": "None", "0": "None", "n": "N_S", "e": "E_W", "b": "Both"}


def lin_board(text):
    """(pbn, vul, auction) of one .lin board dealt by North, or None."""
    i = text.find("md|3")
    if i < 0:
        return None
    hands = text[i + 4: text.find("|", i + 3)].split(",")
    if len(hands) < 3:
        return None
    deal = {}
    for seat, h in zip("SWN", hands[:3]):
        suits, cur = {"S": "", "H": "", "D": "", "C": ""}, None
        for ch in h.upper():
            if ch in suits:
                cur = ch
            elif cur and ch in "AKQJT98765432":
                suits[cur] += ch
            else:
                return None
        if sum(len(v) for v in suits.values()) != 13:
            return None
        deal[seat] = suits
    east = {s: "".join(r for r in "AKQJT98765432" if all(r not in deal[x][s] for x in "SWN")) for s in "SHDC"}
    if sum(len(v) for v in east.values()) != 13:
        return None
    deal["E"] = east
    order = "AKQJT98765432"
    pbn = "N:" + " ".join(".".join("".join(sorted(deal[x][s], key=order.index)) for s in "SHDC") for x in "NESW")
    j = text.find("sv|")
    vul = LIN_VUL.get(text[j + 3: j + 4].lower()) if j >= 0 else None
    if vul is None:
        return None
    calls = []
    k = text.find("|mb|")
    while k >= 0:
        e = text.find("|", k + 4)
        c = text[k + 4: e if e >= 0 else None].strip().rstrip("!").lower()
        if c in ("p", "pass"):
            calls.append("p")
        elif c in ("d", "x", "dbl"):
            calls.append("d")
        elif c in ("r", "xx", "rdbl"):
            calls.append("r")
        elif len(c) == 2 and c[0] in "1234567" and c[1] in "cdhsn":
            calls.append(c[0] + c[1].upper())
        else:
            return None
        k = text.find("|mb|", k + 4) if e >= 0 else -1
    if len(calls) < 4:
        return None
    return pbn, vul, " ".join(calls)


def scan_dirs(dirs):
    from collections import Counter
    rows, files = Counter(), 0
    for d in dirs:
        try:
            for f in os.scandir(d):
                if f.name.endswith(".lin"):
                    files += 1
                    with open(f.path, "rb") as fh:
                        raw = fh.read()
                    if b"md|3" in raw:
                        b = lin_board(raw.decode("utf-8", "replace"))
                        if b:
                            rows[b] += 1
        except OSError:
            pass
    return files, rows


def stage_scan(a):
    import multiprocessing as mp
    from collections import Counter
    import polars as pl
    t0 = time.time()
    dirs = sorted(e.path for e in os.scandir(RAW) if e.is_dir(follow_symlinks=False))
    random.Random(20261119).shuffle(dirs)
    dirs = dirs[: int(len(dirs) * a.share)]
    if a.limit_dirs:
        dirs = dirs[: a.limit_dirs]
    print(f"{len(dirs):,} player folders to read ({time.time() - t0:.0f} s to list)", flush=True)
    chunks = [dirs[i: i + 20] for i in range(0, len(dirs), 20)]
    total, files, done = Counter(), 0, 0
    with mp.Pool(a.workers) as pool:
        for n, rows in pool.imap_unordered(scan_dirs, chunks):
            files += n
            total.update(rows)
            done += 1
            if done % 100 == 0 or done == len(chunks):
                dt = time.time() - t0
                print(f"{done * 20:,} folders, {files:,} files, {files / dt:,.0f} files/s, {len(total):,} North rows, "
                      f"{sum(total.values()):,} tables, {dt / 60:.1f} min", flush=True)
    os.makedirs(a.out, exist_ok=True)
    keys = list(total)
    pl.DataFrame({"PBN": [k[0] for k in keys], "Vul": [k[1] for k in keys], "auction": [k[2] for k in keys],
                  "n": [total[k] for k in keys]}).write_parquet(os.path.join(a.out, "lin_north.parquet"))
    print(f"wrote {len(keys):,} rows on {len({k[0] for k in keys}):,} deals", flush=True)


def dd_of(pbn):
    from endplay.dds import calc_dd_table, par
    from endplay.types import Deal, Denom, Player, Vul
    table = calc_dd_table(Deal(pbn))
    players = {"N": Player.north, "E": Player.east, "S": Player.south, "W": Player.west}
    denoms = {"S": Denom.spades, "H": Denom.hearts, "D": Denom.diamonds, "C": Denom.clubs, "N": Denom.nt}
    tricks = [int(table[denoms[s], players[d]]) for d in SEATS for s in "SHDCN"]
    vuls = {"None": Vul.none, "N_S": Vul.ns, "E_W": Vul.ew, "Both": Vul.both}
    return pbn, tricks, {v: int(par(table, vuls[v], Player.north).score) for v in vuls}


def stage_dd(a):
    import multiprocessing as mp
    import polars as pl
    t0 = time.time()
    src = pl.read_parquet(os.path.join(a.out, "lin_north.parquet"))
    out = os.path.join(a.out, "lin_north_dd.parquet")
    have = pl.read_parquet(out) if os.path.isfile(out) else None
    seen = set(have["PBN"].to_list()) if have is not None else set()
    todo = [p for p in src["PBN"].unique().to_list() if p not in seen]
    print(f"{len(todo):,} deals to solve, {len(seen):,} already done", flush=True)
    rows = []

    def save():
        nonlocal have, rows
        if rows:
            new = pl.DataFrame({"PBN": [r[0] for r in rows], "dd": [r[1] for r in rows],
                                **{f"par_{v}": [r[2][v] for r in rows] for v in ("None", "N_S", "E_W", "Both")}},
                               schema_overrides={"dd": pl.List(pl.UInt8)})
            have = new if have is None else pl.concat([have, new])
            have.write_parquet(out)
            rows = []

    with mp.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(dd_of, todo, chunksize=50), 1):
            rows.append(r)
            if i % 20000 == 0 or i == len(todo):
                save()
                dt = time.time() - t0
                print(f"{i:,} of {len(todo):,} deals, {i / dt:.0f} a second, {dt / 60:.1f} min", flush=True)


def stage_build(a):
    import polars as pl
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from extract_bbo import MAX_CALLS, OUT, VUL_BITS, parse_auction, parse_pbn
    df = pl.read_parquet(os.path.join(a.out, "lin_north.parquet")).join(
        pl.read_parquet(os.path.join(a.out, "lin_north_dd.parquet")), on="PBN", how="inner").sort(["PBN", "Vul", "auction"])
    df = df.with_columns((pl.col("PBN").hash(seed=20261117) % 47 == 0).alias("test"))
    n = df.height
    owner = np.zeros((n, 52), np.uint8)
    calls = np.zeros((n, MAX_CALLS), np.uint8)
    n_calls = np.zeros(n, np.uint8)
    ok = np.zeros(n, bool)
    cache_p, cache_a = {}, {}
    for i, (p, auc) in enumerate(zip(df["PBN"].to_list(), df["auction"].to_list())):
        if p not in cache_p:
            cache_p[p] = parse_pbn(p)
        if auc not in cache_a:
            cache_a[auc] = parse_auction(auc)
        o, t = cache_p[p], cache_a[auc]
        if o is not None and t is not None:
            owner[i], calls[i, : len(t)], n_calls[i], ok[i] = o, t, len(t), True
    vul_names = df["Vul"].to_list()
    vul = np.array([VUL_BITS[v] for v in vul_names], np.uint8)
    par = np.array([row[f"par_{v}"] for row, v in zip(df.select([f"par_{v}" for v in VUL_BITS]).iter_rows(named=True), vul_names)], np.int16)
    dd = np.array(df["dd"].to_list(), np.uint8)
    count = df["n"].to_numpy().astype(np.uint32)
    test = df["test"].to_numpy()
    print(f"{n:,} rows, {int(ok.sum()):,} usable, bad auctions or deals {int((~ok).sum()):,}; "
          f"deals {len(cache_p):,}; held out rows {int((test & ok).sum()):,}", flush=True)
    north = {"owner": owner, "dealer": np.zeros(n, np.uint8), "vul": vul, "calls": calls, "n_calls": n_calls, "dd": dd,
             "par_ns": par, "count": count}
    for name, sel in (("train", ok & ~test), ("test", ok & test)):
        part = {k: v[sel] for k, v in north.items()}
        np.savez(os.path.join(OUT, f"bbo_north_{name}.npz"), **part)
        old = dict(np.load(os.path.join(OUT, f"bbo_{name}.npz")))
        both = {k: np.concatenate([old[k], part[k]]) for k in old}
        np.savez(os.path.join(OUT, f"bbo4_{name}.npz"), **both)
        d = both["dealer"]
        print(f"bbo4_{name}.npz: {len(d):,} rows, by dealer N,E,S,W {[int((d == s).sum()) for s in range(4)]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["scan", "dd", "build"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--share", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=56, help="processes; Windows allows at most 61")
    ap.add_argument("--limit-dirs", type=int, default=0)
    a = ap.parse_args()
    {"scan": stage_scan, "dd": stage_dd, "build": stage_build}[a.stage](a)
