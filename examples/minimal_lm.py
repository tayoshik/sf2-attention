#!/usr/bin/env python3
# Copyright (c) 2026 tayoshik. All rights reserved.
# Licensed under the SF2 Commercial Source License 1.0 (see LICENSE).
"""Minimal example: SF2Attention as the attention sublayer of a small pre-LN causal language model.

Toy task (key-value recall): a random token sequence contains one (KEY, value) pair near the start; the
sequence ends with KEY and the model must predict the value. The pair lies far outside the local window,
so only a far-access path can reach it. This script demonstrates integration and training mechanics on
CPU. With the default 200 steps the model does not solve the task (recall stays at chance); it is not a
benchmark, and no result is claimed.

Run:  python examples/minimal_lm.py [--steps 200] [--local-only]
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from sf2 import SF2Attention

VOCAB, KEY, T = 64, 1, 128          # token 1 is the KEY marker; values / filler use tokens 2..63


class Block(nn.Module):
    def __init__(self, d_model, n_heads, **sf2_kw):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = SF2Attention(d_model, n_heads, **sf2_kw)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        a, _ = self.attn(self.ln1(x))
        x = x + a
        return x + self.ffn(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=2, **sf2_kw):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Embedding(T, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, **sf2_kw) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def batch(n, g):
    x = torch.randint(2, VOCAB, (n, T), generator=g)
    pos = torch.randint(0, 16, (n,), generator=g)              # pair within the first 17 positions
    val = torch.randint(2, VOCAB, (n,), generator=g)
    r = torch.arange(n)
    x[r, pos], x[r, pos + 1], x[:, -1] = KEY, val, KEY
    return x, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--local-only", action="store_true", help="disable every SF2 mechanism (window only)")
    args = ap.parse_args()
    torch.manual_seed(0)
    flags = dict(stream_regularity=False, peak_cap=False, salience_topk=False, signed_dual=False,
                 state_channel=False) if args.local_only else {}
    model = TinyLM(window=16, topk=16, dropout=0.0, **flags)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(1)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}  local_only={args.local_only}")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = batch(16, g)
        loss = F.cross_entropy(model(x)[:, -1], y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"step {step:4d}  loss {loss.item():.3f}  ({time.time() - t0:.1f}s)")
    model.eval()
    with torch.no_grad():
        x, y = batch(512, torch.Generator().manual_seed(2))
        acc = (model(x)[:, -1].argmax(-1) == y).float().mean().item()
    print(f"recall accuracy on 512 held-out sequences: {acc:.3f} (chance {1 / (VOCAB - 2):.3f})")


if __name__ == "__main__":
    main()
