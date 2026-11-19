"""Are the model's probabilities honest? Held-out states, sampled sets, predicted probability against what was true.

  python eval_sets.py --run /data/runs/v1 [--test-file bbo_test.npz --states 3000 --samples 256 --south-only]

Reports, for states at a random point of the auction and for the same hands before any call:
  - honor locations: reliability table (predicted probability vs. how often true), Brier score
  - suit lengths: error of the expected length, probability given to the true length
  - joint events (an honor at a seat AND a length condition at a seat): reliability, against treating them as independent
  - the final contract: predicted chance of making at double dummy vs. how often it makes
  - agreement between the direct DD output and the average over sampled sets
"""
import argparse
import os

import numpy as np
import torch

from model import AuctionModel, sample_sets, view_inputs, view_targets
from scoring import SCORE
from train import load

TOK_STRAIN_TO_DD = [3, 2, 1, 0, 4]          # call tokens use C,D,H,S,N; the DD table uses S,H,D,C,N


def final_contract(calls, dealer):
    """(level, DD strain, declarer seat, doubled) of a finished auction, or None when passed out."""
    last, dbl, first = None, 0, {}
    for i, t in enumerate(calls):
        seat = (dealer + i) % 4
        if t >= 4:
            last, dbl = (t, seat), 0
            first.setdefault((seat % 2, (t - 4) % 5), seat)
        elif t == 2:
            dbl = 1
        elif t == 3:
            dbl = 2
    if last is None:
        return None
    t, seat = last
    return (t - 4) // 5 + 1, TOK_STRAIN_TO_DD[(t - 4) % 5], first[(seat % 2, (t - 4) % 5)], dbl


def reliability(p, y, bins=10):
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    b = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for i in range(bins):
        m = b == i
        if m.any():
            rows.append((i / bins, (i + 1) / bins, int(m.sum()), p[m].mean(), y[m].mean()))
    return rows


def show_rel(title, p, y):
    out = [f"{title}  (Brier {np.mean((np.asarray(p) - np.asarray(y)) ** 2):.4f}, {len(p):,} events)",
           f"  {'predicted':<12}{'events':>10}{'mean predicted':>16}{'came true':>12}"]
    for lo, hi, n, mp, my in reliability(p, y):
        out.append(f"  {lo:.1f}-{hi:.1f}{'':<5}{n:>10,}{mp:>15.1%}{my:>12.1%}")
    return "\n".join(out)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="/data")
    ap.add_argument("--test-file", default="bbo_test.npz")
    ap.add_argument("--states", type=int, default=3000)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--south-only", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(a.run, "model.pt"), map_location=dev, weights_only=False)
    model = AuctionModel(**ck["cfg"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    lookup = torch.tensor(ck["par_lookup"], device=dev)
    D = load(os.path.join(a.data, a.test_file), dev)
    rng = np.random.default_rng(a.seed)
    N = D["n_calls"].shape[0]
    R = min(a.states, N)
    rows = torch.tensor(np.sort(rng.choice(N, R, replace=False)), device=dev)
    view = torch.full((R,), 2, device=dev) if a.south_only else torch.tensor(rng.integers(0, 4, R), device=dev)
    n = D["n_calls"][rows].long()
    t_rand = torch.tensor(rng.random(R), device=dev)
    t_mid = (t_rand * (n + 1)).long().clamp(max=n.max())
    L = int(n.max())
    calls = D["calls"][rows, :L].long()
    feats, callers, lens, hon = view_inputs(D["owner"][rows], D["dealer"][rows], D["vul"][rows], calls)
    dd_rel, _ = view_targets(D["dd"][rows], D["par_ns"][rows], lookup)
    pick = torch.arange(R, device=dev) * 4 + view
    feats, callers, lens, hon, dd_rel = feats[pick], callers[pick], lens[pick], hon[pick], dd_rel[pick].view(R, 4, 5)
    h_all, f = model.encode(feats, calls, callers)
    ar = torch.arange(R, device=dev)
    tricks = torch.arange(14, device=dev, dtype=torch.float32)
    K = a.samples
    out = [f"{a.test_file}, model at step {ck.get('step')}, {R:,} states, {K} sampled sets each"
           + (", South's viewpoint only" if a.south_only else ", all four viewpoints")]

    # final contract of each row, relative to the viewpoint
    cs = [final_contract(c[: int(k)].tolist(), int(d)) for c, k, d in zip(calls.cpu(), n.cpu(), D["dealer"][rows].cpu())]
    vul = D["vul"][rows].cpu().numpy()

    for label, t in (("At a random point of the auction", t_mid), ("Before any call (same hands)", torch.zeros_like(t_mid)),
                     ("After the whole auction", n)):
        h = h_all[ar, t]
        dd_direct = model.dd_head(h).float().view(R, 4, 5, 14).softmax(-1)
        hon_p = torch.zeros(R, 16, 4, device=dev)
        len_p = torch.zeros(R, 4, 3, 14, device=dev)
        joint_p = torch.zeros(R, device=dev)
        same_p = torch.zeros(R, device=dev)
        same_b = torch.zeros(R, device=dev)
        dd_set = torch.zeros(R, 4, 5, 14, device=dev)
        # one joint event per state: honor j at seat r1 AND seat r2 holds at least k cards of suit s
        notmine = (hon > 0).float()
        j_evt = torch.multinomial(notmine + 1e-9, 1).squeeze(1)
        r1 = torch.tensor(rng.integers(1, 4, R), device=dev)
        r2 = torch.tensor(rng.integers(1, 4, R), device=dev)
        s_evt = torch.tensor(rng.integers(0, 4, R), device=dev)
        k_evt = torch.tensor(rng.integers(2, 6, R), device=dev)
        # sets per chunk: the sampler's caches take dec_layers * 2 * 23 * d floats per set; stay inside free GPU memory
        per_set = ck["cfg"]["dec_layers"] * 2 * 23 * ck["cfg"]["d"] * 4 * 2
        free = torch.cuda.mem_get_info()[0] if dev.type == "cuda" else 8 << 30
        chunk = max(1, min(16384, int(0.5 * free / per_set)) // K)
        for i in range(0, R, chunk):
            sl = slice(i, min(i + chunk, R))
            m = sl.stop - sl.start
            rep = lambda x: x[sl].repeat_interleave(K, 0)
            _, sl_lens, sl_hon, _, st = sample_sets(model, rep(h), rep(f), rep(lens[:, :, 0]), rep(hon == 0))
            sl_hon = sl_hon.view(m, K, 16)
            sl_lens = sl_lens.view(m, K, 4, 3)
            hon_p[sl] = torch.nn.functional.one_hot(sl_hon, 4).float().mean(1)
            len_p[sl] = torch.nn.functional.one_hot(sl_lens, 14).float().mean(1)
            dd_set[sl] = model.ddset_head(st).float().view(m, K, 4, 5, 14).softmax(-1).mean(1)
            a_hit = sl_hon[torch.arange(m, device=dev), :, j_evt[sl]] == r1[sl, None]
            b_hit = sl_lens[torch.arange(m, device=dev), :, s_evt[sl], r2[sl] - 1] >= k_evt[sl, None]
            joint_p[sl] = (a_hit & b_hit).float().mean(1)
            # the same honor AND its own holder having k+ cards of the honor's suit: the two parts are related
            c_hit = sl_lens[torch.arange(m, device=dev), :, j_evt[sl] // 4, r1[sl] - 1] >= k_evt[sl, None]
            same_p[sl] = (a_hit & c_hit).float().mean(1)
            same_b[sl] = c_hit.float().mean(1)

        out.append(f"\n=== {label}")
        # honors
        mask = (hon > 0)[:, :, None].expand(-1, -1, 3)
        p = hon_p[:, :, 1:][mask].cpu().numpy()
        y = torch.nn.functional.one_hot(hon, 4)[:, :, 1:][mask].float().cpu().numpy()
        out.append(show_rel("Honor locations (each missing A/K/Q/J at each hidden seat)", p, y))
        top = (hon_p[:, :, 1:].argmax(-1) + 1 == hon)[hon > 0].float().mean()
        out.append(f"  most likely seat is right {float(top):.1%} of the time (one in three by chance)")
        # lengths
        true_len = lens[:, :, 1:]                                                       # (R, suit, 3)
        e_len = (len_p * tricks).sum(-1)
        out.append(f"Suit lengths: expected length off by {float((e_len - true_len).abs().mean()):.3f} cards on average; "
                   f"probability given to the true length {float(len_p.gather(-1, true_len[..., None]).mean()):.1%}")
        # joint
        ya = hon[ar, j_evt] == r1
        yb = lens[ar, s_evt, r2] >= k_evt
        pj = joint_p.cpu().numpy()
        out.append(show_rel("Joint events (an honor at a seat AND that or another seat holding k+ cards of a suit)",
                            pj, (ya & yb).float().cpu().numpy()))
        pa = hon_p[ar, j_evt, r1]
        ge = torch.arange(14, device=dev)[None, :] >= k_evt[:, None]
        pb = (len_p[ar, s_evt, r2 - 1] * ge).sum(-1)
        out.append(f"  Brier if the two parts were treated as independent: "
                   f"{float(((pa * pb) - (ya & yb).float()).pow(2).mean()):.4f}")
        yc = lens[ar, j_evt // 4, r1] >= k_evt
        ys_ = (ya & yc).float()
        out.append(show_rel("Related joint events (an honor at a seat AND that seat holding k+ cards of the honor's suit)",
                            same_p.cpu().numpy(), ys_.cpu().numpy()))
        out.append(f"  Brier if the two parts were treated as independent: {float(((pa * same_b) - ys_).pow(2).mean()):.4f}"
                   f"; mean predicted {float(same_p.mean()):.1%}, independent {float((pa * same_b).mean()):.1%}, true {float(ys_.mean()):.1%}")
        # DD agreement
        e_direct = (dd_direct * tricks).sum(-1)
        e_set = (dd_set * tricks).sum(-1)
        out.append(f"DD tricks: direct output off by {float((e_direct - dd_rel).abs().mean()):.3f}, average over sampled sets off by "
                   f"{float((e_set - dd_rel).abs().mean()):.3f}; the two differ by {float((e_direct - e_set).abs().mean()):.3f}")
        # final contract
        pm, ym, es, ys = [], [], [], []
        ddn = dd_direct.cpu().numpy()
        truth = dd_rel.cpu().numpy()
        vw = view.cpu().numpy()
        for i, c in enumerate(cs):
            if c is None:
                continue
            level, strain, decl, dbl = c
            rel = (decl - vw[i]) % 4
            v = bool(vul[i] & (1 if decl % 2 == 0 else 2))
            sc = SCORE[level - 1, strain, dbl, int(v)]
            pm.append(ddn[i, rel, strain, level + 6:].sum())
            ym.append(float(truth[i, rel, strain] >= level + 6))
            es.append(float((ddn[i, rel, strain] * sc).sum()))
            ys.append(float(sc[truth[i, rel, strain]]))
        out.append(show_rel("Final contract makes at double dummy", pm, ym))
        es, ys = np.array(es), np.array(ys)
        out.append(f"  expected DD score of the final contract off by {np.abs(es - ys).mean():.0f} points on average "
                   f"(mean predicted {es.mean():+.0f}, mean actual {ys.mean():+.0f}, declarer's side)")
    txt = "\n".join(out)
    name = "sets_" + a.test_file.split(".")[0] + ("_south" if a.south_only else "") + ".txt"
    open(os.path.join(a.run, name), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
