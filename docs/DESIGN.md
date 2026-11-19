# SF2 Attention — Design

SF2 is a causal self-attention sublayer. For an input `h` of shape `[B, T, D]` with `H` heads, it returns `out` of the same shape.

Each mechanism below is an independent constructor flag, so that each one can be attributed and ablated separately. This document describes reference implementation version 1.0.0.

## Background

SF2 descends from the SRSM attention, which selected long-range positions by content-independent "renewal events" from a regularity monitor. SF2 keeps the SRSM ideas of regularity, state and sparse selection, but changes three things:
- it makes long-range selection **content-directed and learned**;
- it replaces the prefix-FFT regularity monitor with a **streaming** statistic;
- it adds **signed evidence** and a **gated state** channel.

## F — Streaming regularity and peak cap (`stream_regularity`, `peak_cap`)

Regularity statistic (causal, O(T·D)):

```
Q_t        = w_geom · mean_d (h_t − h_{t−1})²  +  w_freq · mean_d (h_t − EMA_t(h))²
EMA_t(h)   = Σ_{j≤t} (1 − a) a^{t−j} h_j           (a = ema_decay, computed in parallel)
laminar_t  = sigmoid((eps_star − Q_t) / sigma_q)
```

`w_geom` and `w_freq` are learned scalars. `laminar_t` is a soft gate: it multiplies the salience (C) and the state write gate (G).

**Peak cap.** After normalisation, every weight is clipped to magnitude `1 − eta`. The clipped (signed) mass is spread uniformly over the allowed keys that still have room in its direction, and this repeats until no key exceeds the cap. The row sum is preserved. A row is left unchanged only when the cap is infeasible, i.e. when (allowed keys) × (1 − eta) < |row sum|; for a normalised row that means a single allowed key, such as the first position.

Version 1.0.0 also returned the excess to the clipped key itself, so the peak could reach `1 − eta + eta / cnt` (0.975 for two allowed keys with `eta = 0.05`). It also changed the row sum of signed rows with tied or negative peaks. Fixed in 1.0.1.

## C — Content-directed sparse access (`salience_topk`)

Each query position `t` may attend to two sets of keys:
- the **local window**: positions `t − window < j ≤ t`;
- the **top-k far keys**: the `topk` earlier positions outside the window with the highest salience,

```
salience_j = ⟨a, h_j⟩ · laminar_j        (a learned; laminar term only if F is on)
```

The selected far keys receive `salience_j` as an additive bias on their attention scores, so the selector receives gradient through the attention output.

Far-key weight mass is multiplied by a learned gate `sigmoid(far_gate)` (initialised at −4, ≈ 2%), so far access opens as training finds it useful.

With the flag off, the layer is purely local (window only).

## E — Signed, competing evidence (`signed_dual`)

Scores `s = q·kᵀ / √d_head + bias`, restricted to the allowed keys, are normalised without softmax.

```
positive channel  p = (s − min s)₊ / Σ (s − min s)₊           (shifted L1)
signed channel    r = s / Σ |s|                                (can be negative)
weights           w = (1 − m_h) · p + m_h · r,   m_h = sigmoid(mix_h), per head
```

`mix_h` starts at −3 (≈ 5% signed share). With the flag off, only `p` is used.

## G — Gated state channel (`state_channel`)

```
g_t = sigmoid(W_g h_t) · laminar_t        (laminar term only if F is on)
u_t = W_u h_t
m_t = (1 − g_t) m_{t−1} + g_t u_t
out += W_m m_t
```

The recurrence is computed exactly in parallel, as

```
m_t = Σ_{j≤t} g_j u_j Π_{l=j+1..t} (1 − g_l)
```

using cumulative log-keep sums. `W_m` is initialised to zero, so the state channel starts silent.

## Causality

Every path uses only positions `j ≤ t`. The tests perturb future positions and check that earlier outputs are unchanged, for every single-flag configuration and for the full layer.

## Complexity of this implementation

The reference implementation builds dense `[T, T]` masks and score matrices, and applies sparsity by masking. Time and memory are therefore O(T²·D) per layer, as for dense attention, with a larger constant.

An efficient implementation (streaming top-k plus gathered attention, and a recurrent EMA and state) can have the same function at O(T·(window + topk)·D). It is not included in this release.

## Defaults

| Argument | Default |
|---|---|
| `window`, `topk` | 16, 16 |
| `dropout` | 0.1 |
| `eps_star`, `sigma_q` | 0.05, 0.01 |
| `ema_decay` | 0.9 |
| `eta` | 0.05 |
| initial `w_geom`, `w_freq` | 4e-5, 1e-4 |
| initial `far_gate` | −4.0 |
| initial `mix_h` | −3.0 |
| salience weight init | normal, std 0.01 |
| `W_m` init | zero |
