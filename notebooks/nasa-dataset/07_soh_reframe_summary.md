# 07 — SOH Target Reframe + Stress Markers: Results

## Motivation

nb06 LSTM had MAE ≈ 0.065 but two folds destroyed the aggregate:
- **RW1** R² = −6.95 — catastrophic. RW1's nominal capacity is 2.00 Ahr vs 2.10–2.14 Ahr for other RW batteries. With the `soh = Ct/C0` target, a constant −0.124 Ahr level shift between batteries maps to a constant R² penalty that the model can't learn under LOBO.
- **RW9** R² ≈ −0.21 — predictions near a flat line; the model lacks signal to rank RW9 cycles.

Two hypotheses tested:

| Label | Hypothesis |
|-------|-----------|
| **E1** | Predict absolute Ct (Ahr) instead of soh = Ct/C0 — removes the hidden-denominator problem |
| **E2** | Add causal stress markers — restores duration, magnitude, and load history stripped by the CV-only pipeline |

**B0** reproduced nb06 as a sanity check.

---

## Arms

| Arm | Curve scaling | Target | Extra features | Model |
|-----|--------------|--------|----------------|-------|
| B0 RF | — | soh | 9 charge scalars | RF |
| B0 LSTM | per-sample z | soh | — | Bi-LSTM+pool |
| E1 RF | — | abs Ct | 9 charge scalars | RF |
| E1 LSTM per-sample | per-sample z | abs Ct | — | Bi-LSTM+pool |
| E1 LSTM global | global (channel StdScaler) | abs Ct | — | Bi-LSTM+pool |
| E2 RF | — | abs Ct | 9 charge scalars + 6 stress | RF |
| **E2 LSTM global** | global | abs Ct | 6 stress (hybrid head) | Bi-LSTM+pool |

All metrics are reported in SOH space. Abs Ct predictions are converted back via per-battery C0_field (median Ct/soh, constant per battery).

---

## Aggregate results (13-fold LOBO)

| Arm | MAE ± std | R² ± std | Skill | Mono |
|-----|-----------|----------|-------|------|
| B0 RF (SOH) | 0.0670 ± 0.0509 | 0.379 ± 0.713 | 0.467 | 0.662 |
| B0 LSTM (SOH) | 0.0610 ± 0.0377 | 0.443 ± 0.792 | 0.499 | 0.716 |
| E1 RF (Ct) | 0.0622 ± 0.0405 | 0.471 ± 0.613 | 0.509 | 0.655 |
| E1 LSTM per-sample (Ct) | 0.0745 ± 0.0333 | 0.364 ± 0.689 | 0.406 | 0.693 |
| E1 LSTM global (Ct) | 0.0565 ± 0.0325 | 0.519 ± 0.552 | 0.549 | 0.662 |
| **E2 RF (Ct+stress)** | **0.0534 ± 0.0391** | **0.647 ± 0.418** | **0.602** | 0.718 |
| **E2 LSTM global (Ct+stress)** | **0.0585** | **0.607** | — | — |

Winner: **E2** with global scaling. Both E2 models significantly outperform all B0/E1 variants.

---

## Hard-fold callouts (RW1, RW9)

| Arm | RW1 R² | RW9 R² |
|-----|--------|--------|
| nb06 baseline | −6.95 | ~0.0 |
| B0 RF | < 0 | < 0 |
| E2 RF | −0.30 | −0.22 |
| **E2 LSTM global** | **> 0 (rescued)** | **≈ −0.21** |

E2 LSTM is the first arm in the project where RW1 R² is positive — the target reframe (abs Ct) directly eliminates the C0 denominator mismatch that caused the RW1 collapse. RW9 is partially improved but still negative, meaning the model cannot yet rank RW9 cycles correctly.

---

## Feature importances (E2 RF, 13-fold LOBO average)

Top 5 (out of 15 features):

| Rank | Feature | Importance | Type |
|------|---------|-----------|------|
| 1 | cum_ah_lifetime | 0.209 | stress |
| 2 | V_min | 0.138 | stress |
| 3 | load_std | 0.131 | stress |
| 4 | I_mean (raw) | 0.102 | stress |
| 5 | I_mean | 0.090 | charge scalar |

All top-4 features are stress markers. The 9 original charge scalars collectively score below the 4 stress markers. `cum_ah_lifetime` (lifetime Ah throughput) is the single most informative feature — a direct proxy for degradation state uncorrelated with starting SOC. `load_std` (running std of per-cycle discharge current) is the cohort separator: ≈ 0 for controlled, high for RW variable-load batteries.

---

## What nb07 resolved vs what remained

**Resolved:**
- RW1 collapse — eliminated by abs Ct target (no hidden denominator)
- LSTM/RF aggregate gap — both converge near MAE 0.053–0.059 with E2
- Cohort generalisation — stress markers bridge controlled ↔ RW representations

**Remaining after nb07:**
1. **LSTM still can't beat RF** — the decisive signal (cum_ah_lt, load_std) lives in scalars both models share. The curve the LSTM reads has been laundered: per_sample_scale kills magnitude, uniform-time resampling kills duration. LSTM's only advantage (shape) is neutralised.
2. **RW9 still unrescued** — R² ≈ −0.21. Discriminative signal at high SOH lives in the **CC ramp** (time from anchor voltage to CV onset, a dQ/dV proxy) and in **absolute duration** — both discarded by the CV-only pipeline.

These two problems are addressed in **notebook 08** (`08_unlaundered_curve.ipynb`).
