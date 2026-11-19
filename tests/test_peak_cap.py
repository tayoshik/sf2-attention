#!/usr/bin/env python3
# Copyright (c) 2026 tayoshik. All rights reserved.
# Licensed under the SF2 Commercial Source License 1.0 (see LICENSE).
"""Peak cap tests: after cap(), no allowed key exceeds 1 - eta in magnitude whenever the cap is feasible, row
sums are preserved, masked keys stay zero, and rows already under the cap are unchanged.
Run: python -m unittest discover -s tests -v"""
import unittest

import torch

from sf2 import SF2Attention


def layer(eta=0.05):
    return SF2Attention(8, 1, dropout=0.0, eta=eta)


def cap(m, w, mask=None):
    w = torch.as_tensor(w, dtype=torch.float64)
    while w.dim() < 4:
        w = w[None]
    mask = torch.ones_like(w, dtype=torch.bool) if mask is None else mask
    return m.cap(w, mask)


class PeakCap(unittest.TestCase):
    def test_reported_case(self):                       # 1.0.0 returned [0.025, 0.975]
        out = cap(layer(), [0.0, 1.0])[0, 0, 0]
        self.assertTrue(torch.allclose(out, torch.tensor([0.05, 0.95], dtype=torch.float64)))

    def test_one_hot_any_count(self):
        m = layer()
        for cnt in (2, 3, 4, 16, 64, 183):
            w = torch.zeros(cnt, dtype=torch.float64); w[-1] = 1.0
            out = cap(m, w)
            self.assertLessEqual(out.abs().max().item(), 0.95 + 1e-12, cnt)
            self.assertAlmostEqual(out.sum().item(), 1.0, places=12)

    def test_single_key_is_infeasible_and_unchanged(self):
        self.assertEqual(cap(layer(), [1.0]).item(), 1.0)

    def test_signed_ties_preserve_sum(self):
        m = layer()
        for row in ([0.98, -0.98, 0.04], [1.0, -0.95], [1.0, -1.0, 0.0], [-1.0, 0.0, 0.0, 0.0]):
            out = cap(m, row)
            self.assertLessEqual(out.abs().max().item(), 0.95 + 1e-12, row)
            self.assertAlmostEqual(out.sum().item(), sum(row), places=12)

    def test_random_rows(self):
        g = torch.Generator().manual_seed(0)
        for eta in (0.05, 0.2, 0.5):
            m = layer(eta)
            c = 1 - eta
            w = torch.randn(64, 4, 1, 12, generator=g, dtype=torch.float64)
            w[:, 0] = w[:, 0].abs()
            w[:, 0, :, 0] += 20.0                              # strongly peaked rows
            w = w / w.abs().sum(-1, keepdim=True)                # |w| row mass 1, as normalize() produces
            mask = torch.rand(w.shape, generator=g) > 0.3
            mask[..., 0] = True
            out = m.cap(w, mask)
            feasible = mask.sum(-1, keepdim=True) * c >= (w * mask).sum(-1, keepdim=True).abs()
            self.assertLessEqual(((out.abs() - c) * feasible).max().item(), 1e-12)
            self.assertTrue(torch.allclose(out.sum(-1), (w * mask).sum(-1), atol=1e-12))
            self.assertEqual(out.masked_select(~mask).abs().max().item(), 0.0)

    def test_rows_under_cap_unchanged(self):
        w = torch.tensor([0.1, 0.5, 0.4], dtype=torch.float64)
        self.assertTrue(torch.equal(cap(layer(), w)[0, 0, 0], w))

    def test_gradient_flows(self):
        w = torch.tensor([[[[0.0, 0.2, 1.0]]]], dtype=torch.float64, requires_grad=True)
        cap(layer(), w).pow(2).sum().backward()
        self.assertTrue(torch.isfinite(w.grad).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
