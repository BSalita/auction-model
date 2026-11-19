"""Train and evaluate the auction model. See model.py for what it predicts.

  python train.py --data /data --run /data/runs/NAME [--epochs 8 --batch 1024 --d 256 ...]
  python train.py --data /data --run /data/runs/NAME --eval-only     (full held-out evaluation of the saved model)
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import (AuctionModel, N_SLOTS, PAR_OFFSET, set_logits, set_masks, set_values,
                   view_inputs, view_targets)

AMP = False   # mixed precision is slower for this small model (memory-bound)
BUCKETS = ["0 calls", "1-4", "5-8", "9-12", "13+", "auction over"]
METRICS = ["n", "dd_nll", "dd_acc", "dd_mae", "par_nll", "par_mae", "set_nll", "len_nll", "hon_nll",
           "hon_ok", "hon_n", "ddset_nll", "ddset_acc", "parset_mae"]


def load(path, dev, limit=0):
    z = np.load(path)
    out = {}
    for k in z.files:
        a = z[k][:limit] if limit else z[k]
        if a.dtype == np.uint32:
            a = a.astype(np.int64)
        out[k] = torch.from_numpy(a).to(dev)
    return out


def par_vocab(par_ns):
    """Par values seen in training (both signs) and a lookup from any multiple of 10 to the nearest class."""
    v = np.unique(np.concatenate([par_ns, -par_ns]).astype(np.int64))
    grid = np.arange(-PAR_OFFSET, PAR_OFFSET + 1, 10)
    lookup = np.abs(grid[:, None] - v[None, :]).argmin(axis=1)
    return v, lookup


def run_batch(model, D, idx, lookup, par_values, k_dec=2, every_position=False):
    """Losses for the boards idx, from all four viewpoints. Returns per-position tensors."""
    n = D["n_calls"][idx].long()
    L = int(n.max())
    calls = D["calls"][idx, :L].long()
    feats, callers, lens, hon = view_inputs(D["owner"][idx], D["dealer"][idx], D["vul"][idx], calls)
    dd_rel, par_cls = view_targets(D["dd"][idx], D["par_ns"][idx], lookup)
    n4 = n.repeat_interleave(4)
    N = n4.shape[0]
    dev = n.device
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP and dev.type == "cuda"):
        h, f = model.encode(feats, calls.repeat_interleave(4, 0), callers)
        valid = torch.arange(L + 1, device=dev)[None, :] <= n4[:, None]
        seq, pos = valid.nonzero(as_tuple=True)
        hv = h[seq, pos]
        dd_logits = model.dd_head(hv).float().view(-1, 20, 14)
        par_logits = model.par_head(hv).float()
        if every_position:
            ds, dp = seq, pos
        else:
            ds = torch.arange(N, device=dev).repeat(k_dec)
            dp = (torch.rand(N * k_dec, device=dev) * (n4[ds] + 1)).long().clamp(max=L)
        values = set_values(lens, hon)
        len_mask, hon_mask = set_masks(lens, hon)
        st = model.decode(h[ds, dp], f[ds], values[ds])
        ll, hl = set_logits(model, st, len_mask[ds], hon_mask[ds])
        ddset_logits = model.ddset_head(st[:, N_SLOTS]).float().view(-1, 20, 14)
        parset_logits = model.parset_head(st[:, N_SLOTS]).float()

    out = {"seq": seq, "pos": pos, "ds": ds, "dp": dp, "n4": n4}
    out["dd_nll"] = F.cross_entropy(dd_logits.reshape(-1, 14), dd_rel[seq].reshape(-1), reduction="none").view(-1, 20)
    out["par_nll"] = F.cross_entropy(par_logits, par_cls[seq], reduction="none")
    out["len_nll"] = F.cross_entropy(ll.reshape(-1, 14), values[ds, :6].reshape(-1), reduction="none").view(-1, 6)
    hon_t = hon[ds]
    out["hon_valid"] = hon_t > 0
    out["hon_nll"] = F.cross_entropy(hl.reshape(-1, 3), (hon_t - 1).clamp(min=0).reshape(-1),
                                     reduction="none").view(-1, 16) * out["hon_valid"]
    out["ddset_nll"] = F.cross_entropy(ddset_logits.reshape(-1, 14), dd_rel[ds].reshape(-1), reduction="none").view(-1, 20)
    out["parset_nll"] = F.cross_entropy(parset_logits, par_cls[ds], reduction="none")
    if every_position:
        tr = torch.arange(14, device=dev, dtype=torch.float32)
        out["dd_acc"] = (dd_logits.argmax(-1) == dd_rel[seq]).float().mean(1)
        out["dd_mae"] = ((dd_logits.softmax(-1) * tr).sum(-1) - dd_rel[seq]).abs().mean(1)
        true_par = par_values[par_cls]
        out["par_mae"] = ((par_logits.softmax(-1) * par_values).sum(-1) - true_par[seq]).abs()
        out["hon_ok"] = ((hl.argmax(-1) == hon_t - 1) & out["hon_valid"]).float().sum(1)
        out["ddset_acc"] = (ddset_logits.argmax(-1) == dd_rel[ds]).float().mean(1)
        out["parset_mae"] = ((parset_logits.softmax(-1) * par_values).sum(-1) - true_par[ds]).abs()
    return out


def total_loss(o):
    set_nll = o["len_nll"].sum(1) + o["hon_nll"].sum(1)
    parts = {
        "dd": o["dd_nll"].mean(), "par": o["par_nll"].mean(), "set": set_nll.mean(),
        "ddset": o["ddset_nll"].mean(), "parset": o["parset_nll"].mean(),
    }
    loss = parts["dd"] + 0.25 * parts["par"] + parts["set"] / N_SLOTS + parts["ddset"] + 0.25 * parts["parset"]
    return loss, parts


@torch.no_grad()
def evaluate(model, D, lookup, par_values, rows=0, batch=256):
    """Held-out metrics by number of calls seen (rows) and viewpoint seat (N,E,S,W)."""
    model.eval()
    dev = D["n_calls"].device
    total = D["n_calls"].shape[0]
    order = torch.arange(total, device=dev)
    if rows and rows < total:
        order = order[torch.linspace(0, total - 1, rows, device=dev).long()]
    acc = torch.zeros(len(METRICS), len(BUCKETS) * 4, dtype=torch.float64, device=dev)
    for i in range(0, order.shape[0], batch):
        o = run_batch(model, D, order[i:i + batch], lookup, par_values, every_position=True)
        seq, pos = o["seq"], o["pos"]
        bucket = torch.where(pos == 0, 0, ((pos - 1) // 4 + 1).clamp(max=4))
        vals = {
            "n": torch.ones_like(pos, dtype=torch.float32),
            "dd_nll": o["dd_nll"].mean(1), "dd_acc": o["dd_acc"], "dd_mae": o["dd_mae"],
            "par_nll": o["par_nll"], "par_mae": o["par_mae"],
            "set_nll": o["len_nll"].sum(1) + o["hon_nll"].sum(1),
            "len_nll": o["len_nll"].sum(1), "hon_nll": o["hon_nll"].sum(1),
            "hon_ok": o["hon_ok"], "hon_n": o["hon_valid"].float().sum(1),
            "ddset_nll": o["ddset_nll"].mean(1), "ddset_acc": o["ddset_acc"], "parset_mae": o["parset_mae"],
        }
        final = pos == o["n4"][seq]
        for mi, m in enumerate(METRICS):
            v = vals[m].double()
            acc[mi].index_add_(0, bucket * 4 + seq % 4, v)
            acc[mi].index_add_(0, 5 * 4 + (seq % 4)[final], v[final])
    model.train()
    return acc.view(len(METRICS), len(BUCKETS), 4).cpu().numpy()


def table(acc, views, title):
    a = acc[:, :, views].sum(axis=2)
    n = a[0]
    g = {m: a[i] for i, m in enumerate(METRICS)}
    lines = [title,
             f"{'calls seen':<13}{'states':>10}{'DD nll':>8}{'DD acc':>8}{'DD mae':>8}{'par nll':>9}{'par mae':>9}"
             f"{'set nll':>9}{'shape':>8}{'honors':>8}{'hon acc':>9}{'DD|set':>8}{'acc|set':>9}{'par|set':>9}"]
    for b, name in enumerate(BUCKETS):
        if n[b] == 0:
            continue
        lines.append(
            f"{name:<13}{int(n[b]):>10,}{g['dd_nll'][b]/n[b]:>8.3f}{100*g['dd_acc'][b]/n[b]:>7.1f}%{g['dd_mae'][b]/n[b]:>8.3f}"
            f"{g['par_nll'][b]/n[b]:>9.3f}{g['par_mae'][b]/n[b]:>9.1f}{g['set_nll'][b]/n[b]:>9.2f}{g['len_nll'][b]/n[b]:>8.2f}"
            f"{g['hon_nll'][b]/n[b]:>8.2f}{100*g['hon_ok'][b]/max(g['hon_n'][b],1):>8.1f}%{g['ddset_nll'][b]/n[b]:>8.3f}"
            f"{100*g['ddset_acc'][b]/n[b]:>8.1f}%{g['parset_mae'][b]/n[b]:>9.1f}")
    return "\n".join(lines)


def report(acc):
    return "\n\n".join([table(acc, [0, 1, 2, 3], "All four viewpoints"),
                        table(acc, [2], "South's viewpoint (hidden hands are the three robots)"),
                        table(acc, [0, 1, 3], "North, East, West viewpoints")])


def epoch_batches(n_calls, batch, gen):
    """Random batches of boards with similar auction length (less padding). A batch holds at most
    `batch` boards and at most batch*16 encoder positions per viewpoint, so long auctions get smaller batches."""
    N = n_calls.shape[0]
    perm = torch.randperm(N, device=n_calls.device, generator=gen)
    chunk = batch * 50
    budget = batch * 16
    out = []
    for i in range(0, N, chunk):
        p = perm[i:i + chunk]
        ln = n_calls[p].long()
        o = ln.argsort()
        p = p[o]
        ln = (ln[o] + 1).tolist()
        a = 0
        while a < len(ln):
            b = min(a + batch, len(ln))
            while b - a > 1 and ln[b - 1] * (b - a) > budget:
                b = a + max(1, budget // ln[b - 1])
            out.append(p[a:b])
            a = b
    order = torch.randperm(len(out), generator=torch.Generator().manual_seed(int(perm[0])))
    return [out[i] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=float, default=8)
    ap.add_argument("--batch", type=int, default=768)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--enc", type=int, default=6)
    ap.add_argument("--dec", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--k-dec", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-rows", type=int, default=8000)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--test-file", default="bbo_test.npz")
    ap.add_argument("--resume", action="store_true", help="continue from RUN/model.pt")
    ap.add_argument("--resume-frac", type=float, default=0.0, help="share of training already done (old checkpoints)")
    ap.add_argument("--mem-frac", type=float, default=0.82, help="cap on the share of GPU memory this process may hold")
    ap.add_argument("--train-file", default="bbo_train.npz")
    ap.add_argument("--init", default="", help="start from this model.pt (its size and par classes), new schedule")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(a.run, exist_ok=True)
    if dev.type == "cuda":
        # Past the card's memory WSL spills into system memory and a step takes ten times longer: fail instead.
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    log = open(os.path.join(a.run, "train.log"), "a")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    test = load(os.path.join(a.data, a.test_file), dev)
    ckpt_path = os.path.join(a.run, "model.pt")
    if a.eval_only:
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model = AuctionModel(**ck["cfg"]).to(dev)
        model.load_state_dict(ck["model"])
        par_values = torch.tensor(ck["par_values"], dtype=torch.float32, device=dev)
        lookup = torch.tensor(ck["par_lookup"], device=dev)
        acc = evaluate(model, test, lookup, par_values)
        name = "eval" if a.test_file == "bbo_test.npz" else "eval_" + a.test_file.split(".")[0]
        np.save(os.path.join(a.run, name + ".npy"), acc)
        txt = f"{a.test_file}, model at step {ck.get('step')}\n" + report(acc)
        open(os.path.join(a.run, name + ".txt"), "w").write(txt + "\n")
        say(txt)
        return

    train = load(os.path.join(a.data, a.train_file), dev, a.limit)
    if a.init:
        ck = torch.load(a.init, map_location=dev, weights_only=False)
        pv, lk = np.array(ck["par_values"]), np.array(ck["par_lookup"])
        model = AuctionModel(**{**ck["cfg"], "drop": a.drop}).to(dev)
        model.load_state_dict(ck["model"])
    else:
        pv, lk = par_vocab(train["par_ns"].cpu().numpy())
        model = AuctionModel(len(pv), a.d, a.enc, a.dec, a.heads, a.drop).to(dev)
    par_values = torch.tensor(pv, dtype=torch.float32, device=dev)
    lookup = torch.tensor(lk, device=dev)
    n_param = sum(p.numel() for p in model.parameters())
    N = train["n_calls"].shape[0]
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    batches = epoch_batches(train["n_calls"], a.batch, gen)
    steps_per_epoch = len(batches)
    total_steps = int(a.epochs * steps_per_epoch)
    say(f"args {vars(a)}")
    say(f"train rows {N:,}  test rows {test['n_calls'].shape[0]:,}  par classes {len(pv)}  parameters {n_param:,}  "
        f"steps {total_steps:,} ({steps_per_epoch:,} per epoch)")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "emb" in name else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.98), fused=dev.type == "cuda")

    def lr_at(step):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        return a.lr * 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(1, total_steps - a.warmup)))

    def save():
        torch.save({"model": model.state_dict(), "cfg": model.cfg, "par_values": pv.tolist(),
                    "par_lookup": lk.tolist(), "args": vars(a), "step": step, "total_steps": total_steps,
                    "opt": opt.state_dict()}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    step, t0, run = 0, time.time(), {}
    if a.resume:
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        frac = a.resume_frac or ck["step"] / ck["total_steps"]
        step = int(frac * total_steps)
        say(f"resumed from step {ck['step']} of the saved run = {frac:.1%} done; continuing at step {step}")
    step0 = step
    epoch = 0
    while step < total_steps:
        if epoch:
            batches = epoch_batches(train["n_calls"], a.batch, gen)
        for idx in batches:
            if step >= total_steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            o = run_batch(model, train, idx, lookup, par_values, k_dec=a.k_dec)
            loss, parts = total_loss(o)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + float(v.detach())
            if step % 1000 == 0:
                save()
            if step % a.log_every == 0:
                el = time.time() - t0
                say(f"step {step:>6} epoch {step/steps_per_epoch:5.2f} lr {lr_at(step):.2e} "
                    + " ".join(f"{k} {v/a.log_every:.4f}" for k, v in run.items())
                    + f"  {(step-step0)/el:.2f} steps/s  eta {(total_steps-step)/((step-step0)/el)/60:.0f} min"
                    + (f"  gpu peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB" if dev.type == "cuda" else ""))
                run = {}
                if not math.isfinite(float(loss.detach())):
                    raise SystemExit("loss is not finite")
        epoch += 1
        acc = evaluate(model, test, lookup, par_values, rows=a.eval_rows)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        say(f"--- after epoch {epoch} ({(time.time()-t0)/60:.1f} min), {a.eval_rows} held-out rows\n"
            + table(acc, [0, 1, 2, 3], "All four viewpoints") + "\n" + table(acc, [2], "South's viewpoint"))
        save()

    acc = evaluate(model, test, lookup, par_values)
    np.save(os.path.join(a.run, "eval.npy"), acc)
    txt = report(acc)
    open(os.path.join(a.run, "eval.txt"), "w").write(txt + "\n")
    say("=== full held-out evaluation\n" + txt)
    say(json.dumps({"minutes": (time.time() - t0) / 60}))


if __name__ == "__main__":
    main()
