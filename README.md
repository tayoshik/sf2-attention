# SF2 Attention

SF2 is an attention layer from the SRSM (Semantic Renewal Sequential Model) research line. It replaces the softmax over all earlier positions with:
- a local causal window, plus a small set of **content-selected far positions**;
- a softmax-free normalisation that can carry **signed (competing) evidence**;
- a **streaming regularity** statistic and a **gated state channel**.

This package contains the reference implementation of the layer, its tests, and a minimal integration example.

> **License: not open source.** The source is available for a **30-day, non-production evaluation only**. Any other use, **including internal and research use**, requires a paid commercial license. See [LICENSE](LICENSE). Licensing: tayoshik@outlook.jp.

## Status

Experimental research code. This release makes **no claim** that SF2 outperforms standard (softmax, full) attention in accuracy, speed or memory. Evaluation is ongoing, and results will be published separately when final.

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CPU is sufficient for the tests and the example)

## Usage

```python
import torch
from sf2 import SF2Attention

attn = SF2Attention(d_model=256, n_heads=4, window=16, topk=16, dropout=0.1)
h = torch.randn(2, 512, 256)          # [batch, time, d_model]
out, _ = attn(h)                      # causal; same shape as h
```

`SF2Attention` is a drop-in replacement for a causal self-attention sublayer. Use it inside your own pre-LN block: residual + LayerNorm + feed-forward. [examples/minimal_lm.py](examples/minimal_lm.py) shows a complete tiny language model.

### Mechanisms and flags

Each mechanism can be switched off independently for ablation. All are on by default.

| Flag | Mechanism | What it does |
|---|---|---|
| `salience_topk` | content-directed sparse access | each query sees its last `window` positions plus the `topk` most salient earlier positions; salience is learned and also biases the selected scores, so the selector is trained |
| `signed_dual` | signed evidence | mixes a non-negative channel with a signed channel (learned per head), so evidence can be subtracted as well as added |
| `stream_regularity` | streaming regularity | a causal O(T·D) statistic of local change produces a soft "laminar" gate used by selection and state |
| `peak_cap` | peak cap | no single key receives more than (1 − `eta`) of a row's weight mass |
| `state_channel` | gated state | a persistent latent state m_t = (1 − g_t)·m_{t−1} + g_t·u_t, computed exactly in parallel and added to the output |

Other arguments: `window` (16), `topk` (16), `dropout` (0.1), `eps_star`, `sigma_q`, `ema_decay`, `eta`.

[docs/DESIGN.md](docs/DESIGN.md) gives the equations and the initialisation.

## Tests

```bash
pip install torch
python -m unittest discover -s tests -v
```

The tests check three properties:
- causality under future perturbation, for every single-flag configuration and for the full layer;
- that far positions are reachable only through the salience path;
- that gradients reach every mechanism parameter.

## Example

```bash
python examples/minimal_lm.py --steps 200
python examples/minimal_lm.py --steps 200 --local-only   # every SF2 mechanism off (window only)
```

The example demonstrates integration and training mechanics on a toy key-value recall task. With the default 200 steps, neither configuration solves the task (recall stays at chance, about 0.016). It is a wiring demo, not a benchmark.

## Known limitations

- **Dense reference implementation.** Sparse access is realised by masking dense `[T, T]` score matrices, so time and memory are O(T²) per layer. It is slower than optimised softmax attention kernels.
- **Fixed `topk`.** Every position gets exactly `topk` far keys.
- **Query-independent ranking.** Salience is computed per key, so all queries share one far-key ranking.
- **Initialisation matters.** Far-key mass and the signed share start small (≈ 2% and ≈ 5%) and are learned. Learning far access can need more training data than learning local patterns.

## Citation

If you have a commercial license and publish work that uses SF2, please cite:

```
tayoshik. SF2 Attention (reference implementation), version 1.0.0, 2026.
```
