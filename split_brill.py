"""Split brill.npz by date and build a BBO + Brill training mix.

  python split_brill.py [--data /data --test-share 0.2 --bbo-rows 200000 --repeat 8]

  brill_train.npz   the older events
  brill_test.npz    the newest events (about --test-share of the rows; whole dates only)
  mix_train.npz     brill_train repeated --repeat times + a random --bbo-rows of bbo_train
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/data")
ap.add_argument("--test-share", type=float, default=0.2)
ap.add_argument("--bbo-rows", type=int, default=200000)
ap.add_argument("--repeat", type=int, default=8)
a = ap.parse_args()

z = dict(np.load(os.path.join(a.data, "brill.npz")))
n = len(z["dealer"])
dates = np.unique(z["date"])
rows_by_date = np.array([(z["date"] == d).sum() for d in dates])
cum = np.cumsum(rows_by_date) / n
first_test = dates[np.searchsorted(cum, 1 - a.test_share) + 1]
test = z["date"] >= first_test
per_row = [k for k, v in z.items() if getattr(v, "shape", ()) and v.shape[0] == n]
for name, m in (("brill_train", ~test), ("brill_test", test)):
    np.savez(os.path.join(a.data, name + ".npz"), **{k: z[k][m] for k in per_row})
    d = z["date"][m]
    print(f"{name}: {int(m.sum()):,} rows, {len(np.unique(d))} dates {d.min()}..{d.max()}, "
          f"{int(z['count'][m].sum()):,} tables")
# no deal on both sides
key = lambda m: {o.tobytes() for o in z["owner"][m]}
print("deals on both sides:", len(key(~test) & key(test)))

b = np.load(os.path.join(a.data, "bbo_train.npz"))
pick = np.sort(np.random.default_rng(1).choice(len(b["dealer"]), a.bbo_rows, replace=False))
common = [k for k in per_row if k in b.files]
mix = {k: np.concatenate([np.repeat(z[k][~test], a.repeat, axis=0), b[k][pick].astype(z[k].dtype)]) for k in common}
np.savez(os.path.join(a.data, "mix_train.npz"), **mix)
print(f"mix_train: {len(mix['dealer']):,} rows ({int((~test).sum()) * a.repeat:,} Brill, {a.bbo_rows:,} BBO); columns {common}")
