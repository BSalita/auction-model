"""Auction model: from one hand and the auction so far, predict
  - double-dummy tricks for every declarer and strain (20 distributions over 0..13 tricks)
  - the par score (a distribution over score values)
  - the joint distribution of suit lengths and A/K/Q/J locations in the three hidden hands
    (autoregressive: 6 suit lengths, then 16 honors), and DD tricks / par given such a set.

Everything is relative to the viewpoint seat: seat 0 = me, 1 = LHO, 2 = partner, 3 = RHO.
Card index = suit*13 + rank, suits S,H,D,C, ranks A,K,Q,J,T,9..2.
DD cells = declarer (me, LHO, partner, RHO) x strain (S,H,D,C,N).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CALL_TOK = 39          # 0 pad, 1 pass, 2 double, 3 redouble, 4..38 = 1C..7N
MAX_CALLS = 40
N_FEAT = 52 + 56 + 1 + 2 + 4
N_LEN_SLOTS = 6          # S,H,D lengths of LHO and partner; the rest is determined
N_HON = 16               # SA SK SQ SJ HA ... CJ
N_SLOTS = N_LEN_SLOTS + N_HON
TOK_HON0 = 14            # decoder value tokens: 0..13 lengths, 14 mine, 15..17 honor at LHO/partner/RHO
TOK_START = 18
N_DEC_TOK = 19
PAR_OFFSET = 7600        # par scores are multiples of 10 in [-7600, 7600]
HCP_W = torch.tensor([4.0, 3.0, 2.0, 1.0])


class Block(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.heads = heads
        self.drop = drop

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.ff(self.n2(x))

    def step(self, x, cache, max_len):
        """One new token (B,1,C) attending to itself and the cached earlier tokens.
        The cache is (keys, values, count) in buffers allocated once for max_len tokens: growing them with
        torch.cat each step took several times the memory and pushed the sampler out of the GPU."""
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        if cache is None:
            kb = k.new_empty(B, self.heads, max_len, C // self.heads)
            vb = v.new_empty(B, self.heads, max_len, C // self.heads)
            n = 0
        else:
            kb, vb, n = cache
        kb[:, :, n] = k[:, :, 0]
        vb[:, :, n] = v[:, :, 0]
        a = F.scaled_dot_product_attention(q, kb[:, :, : n + 1], vb[:, :, : n + 1])
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.ff(self.n2(x)), (kb, vb, n + 1)


class AuctionModel(nn.Module):
    def __init__(self, n_par, d=256, enc_layers=6, dec_layers=4, heads=8, drop=0.0):
        super().__init__()
        self.cfg = dict(n_par=n_par, d=d, enc_layers=enc_layers, dec_layers=dec_layers, heads=heads, drop=drop)
        self.feat = nn.Sequential(nn.Linear(N_FEAT, d), nn.GELU(), nn.Linear(d, d))
        self.call_emb = nn.Embedding(N_CALL_TOK, d)
        self.seat_emb = nn.Embedding(4, d)
        self.pos_emb = nn.Embedding(MAX_CALLS + 1, d)
        self.enc = nn.ModuleList([Block(d, heads, drop) for _ in range(enc_layers)])
        self.enc_norm = nn.LayerNorm(d)
        self.dd_head = nn.Linear(d, 20 * 14)
        self.par_head = nn.Linear(d, n_par)

        self.dec_ctx = nn.Linear(2 * d, d)
        self.dec_tok = nn.Embedding(N_DEC_TOK, d)
        self.dec_slot = nn.Embedding(N_SLOTS + 1, d)
        self.dec = nn.ModuleList([Block(d, heads, drop) for _ in range(dec_layers)])
        self.dec_norm = nn.LayerNorm(d)
        self.len_head = nn.Linear(d, 14)
        self.hon_head = nn.Linear(d, 3)
        self.ddset_head = nn.Linear(d, 20 * 14)
        self.parset_head = nn.Linear(d, n_par)

    def encode(self, feats, calls, callers):
        """feats (N, N_FEAT), calls (N, L) tokens, callers (N, L) relative seat of each caller.
        Returns states (N, L+1, d): state t has seen the first t calls. Also the hand embedding."""
        f = self.feat(feats)
        L = calls.shape[1]
        pos = self.pos_emb.weight[: L + 1]
        x = torch.cat([f[:, None, :], self.call_emb(calls) + self.seat_emb(callers) + f[:, None, :]], dim=1) + pos
        for b in self.enc:
            x = b(x)
        return self.enc_norm(x), f

    def decode(self, h, f, values):
        """h, f (M, d); values (M, k) decoder value tokens of the first k slots (k <= N_SLOTS).
        Returns states (M, k+1, d): state j predicts slot j; state N_SLOTS has seen the whole set."""
        M, k = values.shape
        start = torch.full((M, 1), TOK_START, dtype=torch.long, device=values.device)
        x = self.dec_tok(torch.cat([start, values], dim=1)) + self.dec_slot.weight[: k + 1]
        x = x + self.dec_ctx(torch.cat([h, f], dim=-1))[:, None, :]
        for b in self.dec:
            x = b(x)
        return self.dec_norm(x)


    def decode_step(self, ctx, tok, slot, caches):
        """Incremental decode: feed one value token at position `slot`; returns that position's state."""
        x = (self.dec_tok(tok) + self.dec_slot.weight[slot] + ctx)[:, None, :]
        new = []
        for i, b in enumerate(self.dec):
            x, c = b.step(x, None if caches is None else caches[i], N_SLOTS + 1)
            new.append(c)
        return self.dec_norm(x)[:, 0], new


# ---------------------------------------------------------------------------------------------
# Building inputs and targets from absolute-seat arrays
# ---------------------------------------------------------------------------------------------

def view_inputs(owner, dealer, vul, calls):
    """owner (B,52) seat 0..3 of each card, dealer (B,), vul (B,) bit0 NS bit1 EW, calls (B,L).
    Returns per (board, viewpoint) tensors flattened to (B*4, ...): feats, callers, lens, hon.
      lens (N, 4 suits, 4 rel seats) suit lengths; hon (N,16) rel seat of each honor (0 = mine)."""
    B, L = calls.shape
    dev = owner.device
    v = torch.arange(4, device=dev)
    rel = (owner[:, None, :].long() - v[None, :, None]) % 4                      # (B,4,52)
    oh = F.one_hot(rel.view(B, 4, 4, 13), 4)                                     # (B,4,suit,rank,seat)
    lens = oh.sum(dim=3)                                                         # (B,4,suit,seat)
    hon = rel.view(B, 4, 4, 13)[..., :4].reshape(B, 4, 16)
    mine = (rel == 0).float()
    mylen = lens[..., 0]
    hcp = (mine.view(B, 4, 4, 13)[..., :4] * HCP_W.to(dev)).sum(dim=(2, 3))
    ns, ew = (vul & 1).float(), ((vul >> 1) & 1).float()
    is_ns = (v % 2 == 0)[None, :]
    we = torch.where(is_ns, ns[:, None], ew[:, None])
    they = torch.where(is_ns, ew[:, None], ns[:, None])
    dealer_rel = (dealer[:, None].long() - v[None, :]) % 4
    feats = torch.cat([
        mine, F.one_hot(mylen, 14).float().view(B, 4, 56), hcp[..., None] / 10.0,
        we[..., None], they[..., None], F.one_hot(dealer_rel, 4).float(),
    ], dim=-1)
    callers = (dealer_rel[:, :, None] + torch.arange(L, device=dev)[None, None, :]) % 4
    N = B * 4
    return feats.view(N, N_FEAT), callers.view(N, L), lens.view(N, 4, 4), hon.view(N, 16)


def view_targets(dd, par_ns, par_lookup):
    """dd (B,20) declarer N,E,S,W x strain; par_ns (B,). Returns dd_rel (B*4,20), par class (B*4,)."""
    B = dd.shape[0]
    dev = dd.device
    v = torch.arange(4, device=dev)
    seat = (v[:, None] + v[None, :]) % 4                                         # (view, rel declarer) -> abs
    dd_rel = dd.view(B, 4, 5).long()[:, seat, :]                                 # (B,4,4,5)
    sign = torch.where(v % 2 == 0, 1, -1)[None, :]
    par_rel = par_ns[:, None].long() * sign
    return dd_rel.reshape(B * 4, 20), par_lookup[(par_rel + PAR_OFFSET) // 10].view(B * 4)


def set_values(lens, hon):
    """Decoder value tokens (N, 22) of the true set: S,H,D lengths of LHO and partner, then 16 honors."""
    return torch.cat([lens[:, :3, 1:3].reshape(-1, 6), TOK_HON0 + hon], dim=1)


def set_masks(lens, hon):
    """Legal choices at each slot given the true earlier slots.
    Returns len_mask (N,6,14) and hon_mask (N,16,3). Every true value is legal by construction."""
    N = lens.shape[0]
    dev = lens.device
    ar = torch.arange(14, device=dev)[None, :]
    rem = 13 - lens[:, :, 0]
    R = torch.full((N, 3), 13, dtype=torch.long, device=dev)                     # cards still to place: LHO, partner, RHO
    masks = []
    for s in range(3):
        a, b, c = lens[:, s, 1], lens[:, s, 2], lens[:, s, 3]
        r = rem[:, s]
        lo = (r - R[:, 1] - R[:, 2]).clamp(min=0)
        hi = torch.minimum(R[:, 0], r)
        masks.append((ar >= lo[:, None]) & (ar <= hi[:, None]))
        lo = (r - a - R[:, 2]).clamp(min=0)
        hi = torch.minimum(R[:, 1], r - a)
        masks.append((ar >= lo[:, None]) & (ar <= hi[:, None]))
        R = R - torch.stack([a, b, c], dim=1)
    len_mask = torch.stack(masks, dim=1)
    oh = F.one_hot(hon.view(N, 4, 4), 4)                                         # (N,suit,rank,seat)
    earlier = oh.cumsum(dim=2) - oh
    cap = lens[:, :, None, 1:] - earlier[..., 1:]                                # (N,suit,rank,3)
    return len_mask, (cap > 0).view(N, 16, 3)


NEG = -1e9


def set_logits(model, states, len_mask, hon_mask):
    """Masked logits of the 22 slots from decoder states (M, >=22, d)."""
    ll = model.len_head(states[:, :N_LEN_SLOTS]).float().masked_fill(~len_mask, NEG)
    hl = model.hon_head(states[:, N_LEN_SLOTS:N_SLOTS]).float().masked_fill(~hon_mask, NEG)
    return ll, hl


@torch.no_grad()
def sample_sets(model, h, f, mylen, myhon, temperature=1.0):
    """Draw one set per row. h, f (M,d) context; mylen (M,4) my suit lengths; myhon (M,16) bool, honor is mine.
    Returns values (M,22) decoder tokens, lens (M,4 suits,3 seats) for LHO/partner/RHO, hon (M,16) rel seat
    (0 mine), logp (M,) the model's log probability of the set, and the decoder state that has seen the set."""
    M = h.shape[0]
    dev = h.device
    ar = torch.arange(14, device=dev)[None, :]
    rem = 13 - mylen
    R = torch.full((M, 3), 13, dtype=torch.long, device=dev)
    lens = torch.zeros((M, 4, 3), dtype=torch.long, device=dev)
    logp = torch.zeros(M, device=dev)
    ctx = model.dec_ctx(torch.cat([h, f], dim=-1))
    state = {"caches": None, "slot": 0}
    values = []

    def feed(tok):
        st, state["caches"] = model.decode_step(ctx, tok, state["slot"], state["caches"])
        state["slot"] += 1
        return st

    def draw(logits, mask):
        lg = logits.float().masked_fill(~mask, NEG)
        x = torch.multinomial((lg / temperature).softmax(-1), 1).squeeze(1)
        return x, F.log_softmax(lg, dim=-1).gather(1, x[:, None]).squeeze(1)

    st = feed(torch.full((M,), TOK_START, dtype=torch.long, device=dev))
    for s in range(3):
        r = rem[:, s]
        lo = (r - R[:, 1] - R[:, 2]).clamp(min=0)
        hi = torch.minimum(R[:, 0], r)
        a, lp = draw(model.len_head(st), (ar >= lo[:, None]) & (ar <= hi[:, None]))
        logp += lp
        values.append(a)
        st = feed(a)
        lo = (r - a - R[:, 2]).clamp(min=0)
        hi = torch.minimum(R[:, 1], r - a)
        b, lp = draw(model.len_head(st), (ar >= lo[:, None]) & (ar <= hi[:, None]))
        logp += lp
        values.append(b)
        st = feed(b)
        abc = torch.stack([a, b, r - a - b], dim=1)
        lens[:, s] = abc
        R = R - abc
    lens[:, 3] = R
    cap = lens.clone()
    hon = torch.zeros((M, 16), dtype=torch.long, device=dev)
    rows = torch.arange(M, device=dev)
    for j in range(16):
        s = j // 4
        mine = myhon[:, j]
        x, lp = draw(model.hon_head(st), (cap[:, s] > 0) | mine[:, None])
        loc = torch.where(mine, 0, x + 1)
        logp += torch.where(mine, 0.0, lp)
        dec = torch.zeros_like(cap[:, s])
        dec[rows, x] = (~mine).long()
        cap[:, s] -= dec
        hon[:, j] = loc
        values.append(TOK_HON0 + loc)
        st = feed(TOK_HON0 + loc)
    return torch.stack(values, dim=1), lens, hon, logp, st
