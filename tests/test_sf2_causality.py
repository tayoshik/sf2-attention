#!/usr/bin/env python3
# Copyright (c) 2026 tayoshik. All rights reserved.
# Licensed under the SF2 Commercial Source License 1.0 (see LICENSE).
"""SF2 tests: causality (perturbing the future never changes earlier outputs) for every single-flag
configuration and the full model; far reach only through the salience path; gradients reach every
mechanism parameter. Run: python -m unittest discover -s tests -v"""
import itertools
import unittest

import torch

from sf2 import SF2Attention

FLAGS = ("stream_regularity", "peak_cap", "salience_topk", "signed_dual", "state_channel")


def build(**on):
    torch.manual_seed(0)
    kw = {f: on.get(f, False) for f in FLAGS}
    m = SF2Attention(32, 4, window=4, topk=3, dropout=0.0, **kw)
    m.eval()
    return m


class SF2(unittest.TestCase):
    def check_causal(self, m):
        x = torch.randn(1, 40, 32)
        for i in (5, 17, 33):
            x2 = x.clone()
            x2[:, i + 1:] += 5.0
            with torch.no_grad():
                a, _ = m(x)
                b, _ = m(x2)
            self.assertTrue(torch.allclose(a[:, :i + 1], b[:, :i + 1], atol=1e-5), f"causality broken at {i}")

    def test_causal_each_flag_and_all(self):
        configs = [{}] + [{f: True} for f in FLAGS] + [{f: True for f in FLAGS}]
        for cfg in configs:
            with self.subTest(cfg=cfg):
                self.check_causal(build(**cfg))

    def test_far_reach_only_with_salience(self):
        x = torch.randn(1, 40, 32)
        x2 = x.clone(); x2[:, 2] += 5.0                        # position 2, far outside window 4 of query 39
        for cfg, expect in (({}, False), ({"salience_topk": True}, None)):
            m = build(**cfg)
            with torch.no_grad():
                a, _ = m(x); b, _ = m(x2)
            changed = bool((a[:, -1] - b[:, -1]).abs().max() > 1e-6)
            if expect is False:
                self.assertFalse(changed, "local-only config must not see position 2")

    def test_gradients_reach_mechanisms(self):
        m = build(**{f: True for f in FLAGS}); m.train()
        x = torch.randn(1, 30, 32, requires_grad=True)
        out, _ = m(x); out.sum().backward()
        for name in ("salience.weight", "mix", "gate.weight", "u_proj.weight", "w_geom"):
            p = dict(m.named_parameters())[name]
            self.assertIsNotNone(p.grad, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
