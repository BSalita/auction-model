"""Paired comparison of scored Brill replays (ben\\scripts\\match\\brill\\score_brill_replay.py) against a baseline.

  elo\\.venv\\Scripts\\python.exe auction-model\\replay_compare.py data\\brill-replay\\am-report old [--split 2026-11-06]

For every other config in OUT_DIR\\boards.parquet: its average, how many auctions differ from the baseline's, and
the difference on the same boards with a 95% interval, for all boards and by scoring. --split also gives the
boards before and from a date (the Brill-tuned model has seen the deals before 2026-11-06). The difference on the
boards whose auction changed shows how much each change is worth.
"""
import pathlib
import sys

import numpy as np
import polars as pl


def pm(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return "   n/a"
    return f"{x.mean():+6.2f} +-{1.96 * x.std(ddof=1) / np.sqrt(len(x)):.2f}"


def main():
    out = pathlib.Path(sys.argv[1])
    base = sys.argv[2]
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else None
    d = pl.read_parquet(out / "boards.parquet")
    key = ["tournament_id", "board"]
    b = d.filter(pl.col("config") == base).select(key + ["scoring", "date", "auction", "contract", "pct_all", "imp_all"])
    lines = [f"baseline {base}: {b.height} boards, {b['pct_all'].mean():.2f}% of the field's matchpoints, {b['imp_all'].mean():+.3f} IMPs a board"]
    for cfg in sorted(set(d["config"].unique()) - {base}):
        c = d.filter(pl.col("config") == cfg).select(key + [pl.col("auction").alias("auction2"), pl.col("contract").alias("contract2"),
                                                            pl.col("pct_all").alias("pct2"), pl.col("imp_all").alias("imp2")])
        j = b.join(c, on=key, how="inner").with_columns((pl.col("pct2") - pl.col("pct_all")).alias("dp"),
                                                        (pl.col("imp2") - pl.col("imp_all")).alias("di"),
                                                        (pl.col("auction") != pl.col("auction2")).alias("changed"),
                                                        (pl.col("contract") != pl.col("contract2")).alias("contract_changed"))
        lines.append(f"\n{cfg}: {j['pct2'].mean():.2f}%, {j['imp2'].mean():+.3f} IMPs; auction differs on {j['changed'].sum()} of {j.height} boards "
                     f"({j['changed'].mean():.1%}), contract on {j['contract_changed'].sum()}")
        parts = [("all boards", j), ("matchpoint events", j.filter(pl.col("scoring") == "MP")), ("IMP events", j.filter(pl.col("scoring") != "MP")),
                 ("boards whose auction changed", j.filter(pl.col("changed")))]
        if split:
            parts += [(f"before {split}", j.filter(pl.col("date") < split)), (f"from {split}", j.filter(pl.col("date") >= split))]
        lines.append(f"  {'boards':<30}{'n':>6}{'percentage points':>22}{'IMPs a board':>20}")
        for name, p in parts:
            lines.append(f"  {name:<30}{p.height:>6}{pm(p['dp']):>22}{pm(p['di']):>20}")
        ch = j.filter(pl.col("changed"))
        lines.append(f"  changed boards: better on {(ch['dp'] > 0).sum()}, worse on {(ch['dp'] < 0).sum()}, same score on {(ch['dp'] == 0).sum()}")
    txt = "\n".join(lines)
    (out / "compare.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
