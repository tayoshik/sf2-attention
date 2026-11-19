# Copyright (c) 2026 tayoshik. All rights reserved.
# Licensed under the SF2 Commercial Source License 1.0 (see LICENSE). Use beyond the evaluation
# grant in LICENSE, including internal and research use, requires a paid commercial license.
"""SF2 attention: a softmax-free, sparse-access attention layer from the SRSM research line.

Drop-in replacement for a causal self-attention sublayer: forward(h) -> (out, None), h of shape
[batch, time, d_model]. Every mechanism is a constructor flag so it can be ablated independently.

F  regularity-preserving stability (flags `stream_regularity`, `peak_cap`)
   Q_t = w_geom * |h_t - h_{t-1}|^2 + w_freq * |h_t - EMA_t(h)|^2   (causal, O(T*D) statistic)
   laminar_t = sigmoid((eps - Q_t) / sigma)  (soft gate used by C and G)
   peak_cap: no single key receives more than (1 - eta) of a row's weight mass (unless the row has one key)
C  content-directed sparse long-range access (flag `salience_topk`)
   every query attends to its local causal window (w) PLUS the top-k most SALIENT earlier positions
   outside the window, salience_j = <a, h_j> (learned) * laminar_j (if F is on). The selected far keys
   also receive the salience as an additive score bias, so the selector is trained.
E  signed / competing evidence (flag `signed_dual`)
   weights = (1 - mix_h) * positive channel (shifted-L1 over allowed keys)
             + mix_h * signed channel (L1 of raw scores, can be negative),  mix_h learned per head
G  dynamic state channel (flag `state_channel`)
   m_t = (1 - g_t) m_{t-1} + g_t u_t,  g_t = sigmoid(W_g h_t) (* laminar_t if F is on),  u_t = W_u h_t
   computed exactly in parallel; output += W_m m_t
Initialisation: signed share sigmoid(-3) ~ 5%; far-key mass gated by sigmoid(far_gate), far_gate = -4
(~2%); salience weights std 0.01; state output projection zero (the state channel starts silent).
All paths are causal by construction (see tests/test_sf2_causality.py).

Note: this is a reference implementation. Sparse access is realised by masking dense [T, T] score
matrices, so time and memory are O(T^2) per layer; it is not an efficient sparse kernel.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SF2Attention(nn.Module):
    def __init__(self, d_model, n_heads, window=16, topk=16, dropout=0.1, stream_regularity=True,
                 peak_cap=True, salience_topk=True, signed_dual=True, state_channel=True,
                 eps_star=0.05, sigma_q=0.01, ema_decay=0.9, eta=0.05):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.window, self.topk = window, topk
        self.flags = dict(stream_regularity=stream_regularity, peak_cap=peak_cap, salience_topk=salience_topk,
                          signed_dual=signed_dual, state_channel=state_channel)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.eps_star, self.sigma_q, self.ema_decay, self.eta = eps_star, sigma_q, ema_decay, eta
        self.w_geom = nn.Parameter(torch.tensor(4e-5))
        self.w_freq = nn.Parameter(torch.tensor(1e-4))
        self.salience = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.salience.weight, std=0.01)          # v1 (dev fix): small initial salience bias
        self.far_gate = nn.Parameter(torch.tensor(-4.0))         # v1 (dev fix): far mass starts at ~2%, learned
        self.mix = nn.Parameter(torch.full((n_heads,), -3.0))   # v1 (dev fix): signed share starts at ~5%
        self.gate = nn.Linear(d_model, 1)
        self.u_proj = nn.Linear(d_model, d_model, bias=False)
        self.m_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.m_proj.weight)                      # state channel starts silent
        self.last = {}

    # ---------------------------------------------------------------- F
    def laminar(self, h):
        B, T, D = h.shape
        dh = torch.zeros_like(h)
        dh[:, 1:] = h[:, 1:] - h[:, :-1]
        # causal EMA via exact parallel form: ema_t = sum_j (1-a) a^(t-j) h_j, a = ema_decay
        a = self.ema_decay
        t = torch.arange(T, device=h.device, dtype=h.dtype)
        w = torch.tril((a ** (t[:, None] - t[None, :]).clamp(min=0)) * (1 - a))
        ema = torch.einsum("ts,bsd->btd", w, h)
        q = self.w_geom * dh.pow(2).mean(-1) + self.w_freq * (h - ema).pow(2).mean(-1)
        return torch.sigmoid((self.eps_star - q) / self.sigma_q), q

    # ---------------------------------------------------------------- C
    def allowed_mask(self, h, lam):
        B, T, _ = h.shape
        i = torch.arange(T, device=h.device)
        causal = i[None, :] <= i[:, None]
        local = causal & (i[None, :] > i[:, None] - self.window)
        if not self.flags["salience_topk"]:
            return local[None, None].expand(B, 1, T, T), None, None
        sal = self.salience(h).squeeze(-1)                       # [B, T]
        if lam is not None:
            sal = sal * lam
        far = causal & ~local                                     # [T, T]
        cand = sal[:, None, :].expand(B, T, T).masked_fill(~far[None], float("-inf"))
        k = min(self.topk, T)
        top = cand.topk(k, dim=-1).indices                         # [B, T, k]
        sel = torch.zeros(B, T, T, dtype=torch.bool, device=h.device).scatter_(-1, top, True) & far[None]
        return (local[None] | sel)[:, None], (sal[:, None, None, :] * sel[:, None].float()), sel[:, None]

    # ---------------------------------------------------------------- E
    def normalize(self, scores, mask, far=None):
        s = scores.masked_fill(~mask, float("inf"))
        mn = s.min(-1, keepdim=True).values
        pos = (scores - mn).clamp(min=0).masked_fill(~mask, 0.0)
        scale = None
        if far is not None:   # far keys enter with a learned gate (v1): mass multiplied by sigmoid(far_gate)
            scale = torch.where(far, torch.sigmoid(self.far_gate), torch.ones((), dtype=scores.dtype))
            pos = pos * scale
        pos = pos / (pos.sum(-1, keepdim=True) + 1e-6)
        if not self.flags["signed_dual"]:
            return pos
        raw = scores.masked_fill(~mask, 0.0)
        if scale is not None:
            raw = raw * scale
        signed = raw / (raw.abs().sum(-1, keepdim=True) + 1e-6)
        m = torch.sigmoid(self.mix)[None, :, None, None]
        return (1 - m) * pos + m * signed

    def cap(self, w, mask):
        """Clip |w| to c = 1 - eta per row and move the clipped (signed) mass to the allowed keys that still have
        room in its direction, repeating until no key exceeds c; the row sum is preserved. A row is left
        unchanged when the cap is infeasible, i.e. when (allowed keys) * c < |row sum| (for a normalised row: a
        single allowed key); for every feasible row a receiver always exists.
        1.0.0 also returned the clipped mass to the clipped key itself, so the peak could reach c + eta / cnt."""
        if not self.flags["peak_cap"]:
            return w
        c = 1 - self.eta
        m = mask.to(w.dtype)
        w = w * m
        feasible = m.sum(-1, keepdim=True) * c >= w.sum(-1, keepdim=True).abs()
        for _ in range(2 * w.shape[-1]):
            over = (w.abs() - c).clamp(min=0) * m * feasible
            over = over * (over > torch.finfo(w.dtype).eps)              # ignore rounding residue at the cap
            if not bool((over > 0).any()):
                break
            clipped = w - torch.sign(w) * over
            moved = (torch.sign(w) * over).sum(-1, keepdim=True)
            recv = m * (torch.sign(moved) * clipped < c).to(w.dtype)
            n = recv.sum(-1, keepdim=True)
            w = torch.where(n > 0, clipped + moved / n.clamp(min=1) * recv, w)
        return w

    # ---------------------------------------------------------------- G
    def state(self, h, lam):
        g = torch.sigmoid(self.gate(h)).squeeze(-1)               # [B, T]
        if lam is not None:
            g = g * lam
        u = self.u_proj(h)
        logkeep = torch.log1p(-g.clamp(max=1 - 1e-6))
        L = torch.cumsum(logkeep, dim=1)                         # L_t = sum_{l<=t} log(1-g_l)
        T = h.shape[1]
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=h.device))
        expo = (L[:, :, None] - L[:, None, :]).masked_fill(~causal[None], float("-inf"))
        W = torch.exp(expo) * g[:, None, :]                       # W[t, j] = g_j prod_{l=j+1..t}(1-g_l)
        return self.m_proj(torch.einsum("btj,bjd->btd", W, u))

    def forward(self, h):
        B, T, D = h.shape
        lam = q = None
        if self.flags["stream_regularity"]:
            lam, q = self.laminar(h)
        mask, bias, far = self.allowed_mask(h, lam)
        qh = self.q_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        kh = self.k_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        vh = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = qh @ kh.transpose(-2, -1) * self.d_head ** -0.5
        if bias is not None:
            scores = scores + bias
        w = self.cap(self.normalize(scores, mask, far), mask)
        out = (self.drop(w) @ vh).transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out)
        if self.flags["state_channel"]:
            out = out + self.state(h, lam)
        self.last = {"laminar_mean": None if lam is None else float(lam.mean().detach())}
        return out, None
