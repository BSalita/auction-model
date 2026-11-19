"""Extract distinct (deal, auction) pairs from the BBO robot file into compact arrays.

Reads data\\bbo_mldf_augmented.parquet of the BBO folder, writes bbo_{train,test}.npz to the data folder (paths.py
names both folders). One row per distinct deal + dealer + vul + auction:
  owner   (N, 52) uint8  seat holding each card, 0..3 = N,E,S,W; card index = suit*13 + rank,
                         suits S,H,D,C, ranks A,K,Q,J,T,9..2
  dealer  (N,)    uint8  0..3 = N,E,S,W
  vul     (N,)    uint8  bit 0 = NS vulnerable, bit 1 = EW vulnerable
  calls   (N, L)  uint8  0 pad, 1 pass, 2 double, 3 redouble, 4 + (level-1)*5 + strain (C,D,H,S,N)
  n_calls (N,)    uint8
  dd      (N, 20) uint8  double-dummy tricks, declarer N,E,S,W x strain S,H,D,C,N
  par_ns  (N,)    int16  par score for North-South
  count   (N,)    uint32 number of tables with this auction
Held-out deals: PBN.hash(seed=20261117) % 47 == 0 (same rule as the real-GIB bidding networks).
"""
import os
import sys
import time

import numpy as np
import polars as pl

import paths

SRC = os.path.join(paths.BBO, "data", "bbo_mldf_augmented.parquet")
OUT = paths.DATA
SEATS = "NESW"
DD_STRAINS = "SHDCN"
RANKS = "AKQJT98765432"
MAX_CALLS = 40
VUL_BITS = {"None": 0, "N_S": 1, "E_W": 2, "Both": 3}

CALL_TOKEN = {"p": 1, "d": 2, "r": 3}
for lvl in range(1, 8):
    for si, s in enumerate("CDHSN"):
        CALL_TOKEN[f"{lvl}{s}"] = 4 + (lvl - 1) * 5 + si


def parse_pbn(pbn):
    o = np.full(52, 255, np.uint8)
    first = SEATS.index(pbn[0])
    hands = pbn[2:].split(" ")
    if len(hands) != 4:
        return None
    for i, h in enumerate(hands):
        suits = h.split(".")
        if len(suits) != 4:
            return None
        seat = (first + i) % 4
        for si, suit in enumerate(suits):
            for ch in suit:
                r = RANKS.find(ch)
                if r < 0:
                    return None
                o[si * 13 + r] = seat
    if (o == 255).any() or not all((o == s).sum() == 13 for s in range(4)):
        return None
    return o


def parse_auction(a):
    """Token list, or None when the auction is not a complete legal-looking auction."""
    toks = []
    last_bid = 0
    for c in a.split(" "):
        t = CALL_TOKEN.get(c)
        if t is None:
            return None
        if t >= 4:
            if t <= last_bid:
                return None
            last_bid = t
        toks.append(t)
    n = len(toks)
    if n < 4 or n > MAX_CALLS:
        return None
    if last_bid == 0:
        return toks if toks == [1, 1, 1, 1] else None
    if toks[-3:] != [1, 1, 1] or toks[-4] == 1 and n == 4:
        return None
    return toks


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    dd_cols = [f"DD_{d}_{s}" for d in SEATS for s in DD_STRAINS]
    lf = (
        pl.scan_parquet(SRC)
        .select(["PBN", "Dealer", "Vul", "bid", "Par_NS", *dd_cols])
        .filter(pl.col("bid").is_not_null() & pl.col("PBN").is_not_null())
        .with_columns(pl.col("bid").list.join(" ").alias("auction"))
        .drop("bid")
        .group_by(["PBN", "Dealer", "Vul", "auction"])
        .agg(pl.len().alias("n"), pl.col("Par_NS").first(), *[pl.col(c).first() for c in dd_cols])
    )
    df = lf.collect(engine="streaming")
    print(f"distinct deal+auction rows: {df.height:,}  ({time.time()-t0:.0f}s)", flush=True)
    df = df.drop_nulls().sort(["PBN", "Dealer", "Vul", "auction"])
    print(f"after dropping nulls: {df.height:,}", flush=True)
    print(df.group_by("Vul").len().sort("Vul"), flush=True)
    df = df.with_columns(
        (pl.col("PBN").hash(seed=20261117) % 47 == 0).alias("test"),
        pl.col("PBN").rle_id().alias("pbn_id"),
    )

    pbns = df["PBN"].unique(maintain_order=True).to_list()
    print(f"distinct deals: {len(pbns):,}", flush=True)
    owners = np.zeros((len(pbns), 52), np.uint8)
    ok_deal = np.zeros(len(pbns), bool)
    for i, p in enumerate(pbns):
        o = parse_pbn(p)
        if o is not None:
            owners[i] = o
            ok_deal[i] = True
    print(f"bad deals: {(~ok_deal).sum():,}  ({time.time()-t0:.0f}s)", flush=True)

    auctions = df["auction"].to_list()
    n = len(auctions)
    calls = np.zeros((n, MAX_CALLS), np.uint8)
    n_calls = np.zeros(n, np.uint8)
    ok = np.zeros(n, bool)
    cache = {}
    for i, a in enumerate(auctions):
        t = cache.get(a)
        if t is None and a not in cache:
            t = parse_auction(a)
            cache[a] = t
        if t is not None:
            calls[i, : len(t)] = t
            n_calls[i] = len(t)
            ok[i] = True
    print(f"distinct auction strings: {len(cache):,}  bad auctions: {(~ok).sum():,}  ({time.time()-t0:.0f}s)", flush=True)

    pbn_id = df["pbn_id"].to_numpy()
    ok &= ok_deal[pbn_id]
    dealer = np.array([SEATS.index(d) if d in SEATS and len(d) == 1 else 255 for d in df["Dealer"].to_list()], np.uint8)
    vul = np.array([VUL_BITS.get(v, 255) for v in df["Vul"].to_list()], np.uint8)
    ok &= (dealer < 4) & (vul < 4)
    dd = df.select(dd_cols).to_numpy().astype(np.uint8)
    ok &= (dd <= 13).all(axis=1)
    par = df["Par_NS"].to_numpy().astype(np.int16)
    count = df["n"].to_numpy().astype(np.uint32)
    test = df["test"].to_numpy()
    print(f"usable rows: {ok.sum():,} of {n:,}", flush=True)

    for name, sel in (("train", ok & ~test), ("test", ok & test)):
        np.savez(
            os.path.join(OUT, f"bbo_{name}.npz"),
            owner=owners[pbn_id[sel]], dealer=dealer[sel], vul=vul[sel], calls=calls[sel],
            n_calls=n_calls[sel], dd=dd[sel], par_ns=par[sel], count=count[sel],
        )
        print(f"{name}: {sel.sum():,} rows, {len(np.unique(pbn_id[sel])):,} deals, "
              f"mean calls {n_calls[sel].mean():.2f}, max {n_calls[sel].max()}", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
