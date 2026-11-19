"""Raw GPU speed and transformer-block speed under different precisions."""
import time

import torch
import torch.nn as nn

from model import Block

dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def timed(name, fn, reps=10):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    ms = (time.time() - t) / reps * 1000
    print(f"{name:<44}{ms:8.1f} ms", flush=True)
    return ms


for dt in (torch.float32, torch.bfloat16, torch.float16):
    a = torch.randn(8192, 8192, device=dev, dtype=dt)
    ms = timed(f"matmul 8192^2 {dt}", lambda: a @ a)
    print(f"   {2*8192**3/ms/1e9:.1f} TFLOP/s")

x = torch.randn(4096, 23, 256, device=dev)
blocks = nn.Sequential(*[Block(256, 8, 0.0) for _ in range(4)]).to(dev)


def run(dtype, bw):
    def f():
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            y = blocks(x)
        if bw:
            y.float().mean().backward()
    return f


for dtype in (None, torch.bfloat16, torch.float16):
    timed(f"4 blocks fwd {dtype}", run(dtype, False))
    timed(f"4 blocks fwd+bwd {dtype}", run(dtype, True))

lin = nn.Linear(256, 1024).to(dev)
x2 = torch.randn(94208, 256, device=dev)
for dtype in (None, torch.bfloat16, torch.float16):
    def f():
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            return lin(x2)
    timed(f"linear 256->1024 on 94k rows {dtype}", f, 50)
ln = nn.LayerNorm(256).to(dev)
timed("layernorm 94k x 256", lambda: ln(x2), 50)
g = nn.GELU()
x3 = torch.randn(94208, 1024, device=dev)
timed("gelu 94k x 1024", lambda: g(x3), 50)
cb = torch.compile(blocks)
for dtype in (None, torch.bfloat16):
    timed(f"compiled 4 blocks fwd+bwd {dtype}", run(dtype, True))
    def fc():
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            y = cb(x)
        y.float().mean().backward()
    timed(f"torch.compile 4 blocks fwd+bwd {dtype}", fc)
