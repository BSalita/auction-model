"""BEN's sampled deals against the auction model's, scored on the true deal (South's turns of held-out auctions).

  python sample_check.py --run /data/runs/v1-brill --ben /data/ben_samples_old.npz [--data /data/brill_test.npz --k 50 256]

`ben_samples.py` made the BEN file (deals from BotBid.sample_hands_for_auction). The same states are given to the
model, which draws K sets each. Both are turned into probabilities the same way (share of the deals) and scored:
honor seats, hidden hands' points, suit lengths, an honor together with length in its suit, and the double-dummy
tricks of the real final contract (BEN: its DD solve of each deal; model: its direct trick output).
Differences are BEN minus model per state, with twice the standard error over boards.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from eval_sets import final_contract
from model import AuctionModel, sample_sets, view_inputs, view_targets
from train import load

HCP = torch.tensor([4.0, 3.0, 2.0, 1.0] * 4)


def source_stats(hon_s, lens_s, valid, hon, lens):
    """hon_s (R,S,16) seat 0..3, lens_s (R,S,4,3), valid (R,S) bool -> per-state measures against the truth."""
    dev = hon.device
    w = valid.float() / valid.float().sum(1, keepdim=True).clamp(min=1)
    hon_p = (F.one_hot(hon_s.long(), 4).float() * w[:, :, None, None]).sum(1)              # (R,16,4)
    len_p = (F.one_hot(lens_s.long(), 14).float() * w[:, :, None, None, None]).sum(1)      # (R,4,3,14)
    hidden = hon > 0
    y = F.one_hot(hon, 4).float()
    m = hidden.float()
    n_h = m.sum(1).clamp(min=1)
    o = {}
    o["honor Brier"] = (((hon_p[:, :, 1:] - y[:, :, 1:]) ** 2).sum(-1) * m).sum(1) / n_h / 3
    o["honor top seat right"] = ((hon_p[:, :, 1:].argmax(-1) + 1 == hon).float() * m).sum(1) / n_h
    o["honor given under 10% is there"] = (hon_p[:, :, 1:], y[:, :, 1:], hidden)
    hcp = HCP.to(dev)
    e_hcp = (hon_p[:, :, 1:] * (hcp * m)[:, :, None]).sum(1)                                # (R,3)
    t_hcp = (y[:, :, 1:] * (hcp * m)[:, :, None]).sum(1)
    o["points of a hidden hand off by"] = (e_hcp - t_hcp).abs().mean(1)
    true_len = lens[:, :, 1:]
    e_len = (len_p * torch.arange(14, device=dev)).sum(-1)
    o["suit length off by (cards)"] = (e_len - true_len).abs().mean((1, 2))
    o["probability of the true length"] = len_p.gather(-1, true_len[..., None]).squeeze(-1).mean((1, 2))
    o["true length never in the deals"] = (len_p.gather(-1, true_len[..., None]).squeeze(-1) == 0).float().mean((1, 2))
    # honor j at seat r AND seat r has 4+ cards of its suit, all hidden honors and seats
    suit = torch.arange(16, device=dev) // 4
    has = F.one_hot(hon_s.long(), 4)[..., 1:].bool()                                        # (R,S,16,3)
    long_ = (lens_s >= 4)[:, :, suit, :]                                                    # (R,S,16,3)
    pj = ((has & long_).float() * w[:, :, None, None]).sum(1)
    yj = y[:, :, 1:].bool() & (true_len >= 4)[:, suit, :]
    o["honor with 4+ cards Brier"] = (((pj - yj.float()) ** 2).sum(-1) * m).sum(1) / n_h / 3
    return o


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ben", required=True)
    ap.add_argument("--data", default="/data/brill_test.npz")
    ap.add_argument("--k", type=int, nargs="+", default=[50, 256])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(3)
    ck = torch.load(os.path.join(a.run, "model.pt"), map_location=dev, weights_only=False)
    model = AuctionModel(**ck["cfg"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    lookup = torch.tensor(ck["par_lookup"], device=dev)
    D = load(a.data, dev)
    B = np.load(a.ben)
    keep = B["n"] > 0
    rows = torch.tensor(B["row"][keep], device=dev).long()
    t = torch.tensor(B["t"][keep], device=dev).long()
    R = len(rows)
    ar = torch.arange(R, device=dev)
    L = int(D["n_calls"][rows].max())
    calls = D["calls"][rows, :L].long()
    feats, callers, lens, hon = view_inputs(D["owner"][rows], D["dealer"][rows], D["vul"][rows], calls)
    dd_rel, _ = view_targets(D["dd"][rows], D["par_ns"][rows], lookup)
    pick = ar * 4 + 2
    feats, callers, lens, hon, dd_rel = feats[pick], callers[pick], lens[pick], hon[pick], dd_rel[pick].view(R, 4, 5)
    h_all, f = [], []
    for i in range(0, R, 1024):
        hh, ff = model.encode(feats[i:i + 1024], calls[i:i + 1024], callers[i:i + 1024])
        h_all.append(hh[torch.arange(hh.shape[0], device=dev), t[i:i + 1024]])
        f.append(ff)
    h, f = torch.cat(h_all), torch.cat(f)

    # the real final contract of each state's board, from South's side of the table
    n_calls = D["n_calls"][rows].cpu()
    dealer = D["dealer"][rows].cpu()
    cs = [final_contract(c[: int(k)].tolist(), int(d)) for c, k, d in zip(calls.cpu(), n_calls, dealer)]
    has_c = torch.tensor([c is not None for c in cs], device=dev)
    level = torch.tensor([c[0] if c else 1 for c in cs], device=dev)
    strain = torch.tensor([c[1] if c else 0 for c in cs], device=dev)
    rel = torch.tensor([(c[2] - 2) % 4 if c else 0 for c in cs], device=dev)
    true_tr = dd_rel[ar, rel, strain]
    made = (true_tr >= level + 6).float()
    tricks14 = torch.arange(14, device=dev, dtype=torch.float32)

    def contract_stats(p_tr):
        e = (p_tr * tricks14).sum(-1)
        pm = (p_tr * (tricks14[None, :] >= (level + 6)[:, None])).sum(-1)
        return {"final contract tricks off by": (e - true_tr).abs(), "final contract makes Brier": (pm - made) ** 2,
                "_pm": pm}

    S = {}
    # BEN
    valid = torch.tensor(np.arange(B["hon"].shape[1])[None, :] < B["n"][keep][:, None], device=dev)
    o = source_stats(torch.tensor(B["hon"][keep], device=dev), torch.tensor(B["lens"][keep], device=dev), valid, hon, lens)
    tr = torch.tensor(B["tricks"][keep], device=dev).long().clamp(min=0)
    p_tr = (F.one_hot(tr, 14).float() * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True)
    o.update(contract_stats(p_tr))
    S["BEN's deals"] = o
    # the model, K sets per state
    dd_direct = model.dd_head(h).float().view(R, 4, 5, 14).softmax(-1)[ar, rel, strain]
    for K in a.k:
        hs, ls = [], []
        chunk = max(1, 8192 // K)
        for i in range(0, R, chunk):
            sl = slice(i, min(i + chunk, R))
            rep = lambda x: x[sl].repeat_interleave(K, 0)
            _, sl_lens, sl_hon, _, _ = sample_sets(model, rep(h), rep(f), rep(lens[:, :, 0]), rep(hon == 0))
            hs.append(sl_hon.view(-1, K, 16))
            ls.append(sl_lens.view(-1, K, 4, 3))
        o = source_stats(torch.cat(hs), torch.cat(ls), torch.ones(R, K, dtype=torch.bool, device=dev), hon, lens)
        o.update(contract_stats(dd_direct))
        S[f"model, {K} sets"] = o

    names = list(S)
    row_id = rows.cpu().numpy()
    turn = torch.tensor([(int(tt) - (2 - int(d)) % 4) // 4 + 1 for tt, d in zip(t.cpu(), dealer)], device=dev)

    def mean_pm(x, mask):
        """Mean and twice the standard error over boards (states of one board are not independent)."""
        x = x[mask].double().cpu().numpy()
        r = row_id[mask.cpu().numpy()]
        _, inv = np.unique(r, return_inverse=True)
        s, c = np.bincount(inv, x), np.bincount(inv)
        g = len(s)
        return x.mean(), 2 * np.sqrt(g / (g - 1) * ((s - x.mean() * c) ** 2).sum()) / c.sum()

    out = [f"{os.path.basename(a.ben)} against {a.run} (step {ck.get('step')}) on {os.path.basename(a.data)}: "
           f"{R:,} South turns on {len(np.unique(row_id)):,} boards; {int((~keep).sum())} turns where BEN returned no deal left out",
           f"BEN deals per turn: mean {B['n'][keep].mean():.1f}, under 15 on {(B['n'][keep] < 15).mean():.1%} of turns; "
           f"sample quality mean {B['quality'][keep].mean():.2f}; sampling {B['secs'][keep].mean():.2f}s a turn"]
    groups = [("All South turns", torch.ones(R, dtype=torch.bool, device=dev))]
    groups += [(f"South's {lab} turn", m) for lab, m in (("first", turn == 1), ("second", turn == 2), ("third or later", turn >= 3))]
    for title, mask in groups:
        out.append(f"\n=== {title}: {int(mask.sum()):,} turns")
        out.append(f"  {'measure':<36}" + "".join(f"{n:>18}" for n in names) + f"{'BEN - ' + names[1]:>30}")
        for key in S[names[0]]:
            if key.startswith("_"):
                continue
            v = [S[n][key] for n in names]
            if isinstance(v[0], tuple):                       # reliability of small probabilities, pooled over events
                cells = []
                for p, y, hid in v:
                    sel = (p < 0.1) & hid[:, :, None] & mask[:, None, None]
                    cells.append(f"{float(y[sel].mean()):>17.1%} ")
                out.append(f"  {key:<36}" + "".join(cells))
                continue
            mk = mask & has_c if key.startswith("final") else mask
            pct = "right" in key or "probability" in key or "never" in key
            fmt = (lambda x: f"{x:>17.1%} ") if pct else (lambda x: f"{x:>17.4f} " if "Brier" in key else f"{x:>17.3f} ")
            d, se = mean_pm(v[0] - v[1], mk)
            dtxt = f"{d:+.1%} +-{se:.1%}" if pct else f"{d:+.4f} +-{se:.4f}"
            out.append(f"  {key:<36}" + "".join(fmt(float(x[mk].mean())) for x in v) + f"{dtxt:>30}")
        # does the final contract make: predicted against true, in three bands
        mk = mask & has_c
        for lo, hi in ((0.0, 0.2), (0.2, 0.8), (0.8, 1.01)):
            cells = []
            for n in names:
                pm = S[n]["_pm"]
                sel = mk & (pm >= lo) & (pm < hi)
                cells.append(f"{int(sel.sum()):>6} {float(pm[sel].mean()) if sel.any() else 0:>4.0%}->{float(made[sel].mean()) if sel.any() else 0:>4.0%} ")
            out.append(f"  {'makes given ' + f'{lo:.0%}-{min(hi, 1):.0%}' + ' (turns, said->true)':<36}" + "".join(f"{c:>18}" for c in cells))
    txt = "\n".join(out)
    path = a.out or os.path.join(a.run, "sample_check_" + os.path.basename(a.ben).split(".")[0] + ".txt")
    open(path, "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
