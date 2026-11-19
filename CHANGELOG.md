# Changelog

## 1.0.1 (2026-11-19)

**Fixed: `peak_cap` could exceed its bound.**

1.0.0 clipped the largest weight of a row to `1 − eta` and then spread the excess over all allowed keys, including the clipped key itself. The peak after the cap was therefore `1 − eta + eta / cnt`, where `cnt` is the number of allowed keys. For example:

| Allowed keys (`eta = 0.05`) | Peak after the cap |
|---|---|
| 2 | 0.975 |
| 16 | 0.953 |
| 1 | 1.0 (the cap was a no-op) |

For signed rows with tied or negative peaks, 1.0.0 also changed the row sum.

1.0.1 does the following:
- Clips every weight to magnitude `1 − eta`.
- Spreads the clipped signed mass only over allowed keys that still have room in its direction, and repeats until no key exceeds the cap.
- Preserves the row sum.
- Leaves a row unchanged when the cap is infeasible: (allowed keys) × (1 − eta) < |row sum|. For a normalised row this means a single allowed key.

Tests: `tests/test_peak_cap.py`.

**Compatibility.** Outputs differ from 1.0.0 only in rows whose peak exceeded `1 − eta`.

The SRSM research results that informed this layer were produced with the 1.0.0 behaviour. Those runs are frozen and are not re-run.
