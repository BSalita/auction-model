"""Where a training step spends its time (one batch, repeated)."""
import sys
import time

import torch

from model import AuctionModel, set_masks, set_values, view_inputs, view_targets
from train import load, par_vocab, run_batch, total_loss

dev = torch.device("cuda")
B = int(sys.argv[1]) if len(sys.argv) > 1 else 512
D = load("/data/bbo_test.npz", dev)
pv, lk = par_vocab(D["par_ns"].cpu().numpy())
par_values = torch.tensor(pv, dtype=torch.float32, device=dev)
lookup = torch.tensor(lk, device=dev)
model = AuctionModel(len(pv)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
idx = torch.randperm(D["n_calls"].shape[0], device=dev)[:B]


def timed(name, fn, reps=10):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        r = fn()
    torch.cuda.synchronize()
    print(f"{name:<28}{(time.time()-t)/reps*1000:8.1f} ms", flush=True)
    return r


n = D["n_calls"][idx].long()
L = int(n.max())
calls = D["calls"][idx, :L].long()
print("batch", B, "max calls", L, "mean", float(n.float().mean()))
feats, callers, lens, hon = timed("view_inputs", lambda: view_inputs(D["owner"][idx], D["dealer"][idx], D["vul"][idx], calls))
timed("view_targets", lambda: view_targets(D["dd"][idx], D["par_ns"][idx], lookup))
timed("set_values+masks", lambda: (set_values(lens, hon), set_masks(lens, hon)))
calls4 = calls.repeat_interleave(4, 0)
values = set_values(lens, hon)


def enc():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.encode(feats, calls4, callers)


h, f = timed("encode forward", enc)
h0 = h[:, 0].detach()
v2 = values.repeat(2, 1)
h2, f2 = h0.repeat(2, 1), f.detach().repeat(2, 1)


def dec():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.decode(h2, f2, v2)


timed("decode forward (k=2)", dec)


def enc_bw():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hh, _ = model.encode(feats, calls4, callers)
    hh.float().mean().backward()


def dec_bw():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s = model.decode(h2, f2, v2)
    s.float().mean().backward()


timed("encode fwd+bwd", enc_bw)
timed("decode fwd+bwd", dec_bw)
o = timed("run_batch forward", lambda: run_batch(model, D, idx, lookup, par_values))


def step():
    o = run_batch(model, D, idx, lookup, par_values)
    loss, _ = total_loss(o)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


timed("full step", step)
print("peak GB", torch.cuda.max_memory_allocated() / 2**30)
