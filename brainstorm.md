# Battery SOH — DL Micro-Clock Brainstorm

*Context doc for phone brainstorm — all background included.*

---

## The Goal

EFREI ML course project. Beat the RF scalar baseline on SOH (capacity_retention) prediction
using DL models trained on raw charge-curve tensors. Deliverable: 15-min presentation + public
GitHub.

---

## The Setup

**Data:** 13 NASA Li-ion batteries.
- 4 **controlled** (B0005–B0007, B0018): 1.5A CC charge from near-empty, every discharge
  cycle retained. 631 samples. Clean, complete degradation.
- 9 **randomized** (RW1, RW9, RW13–RW20): variable charge/discharge profiles, only
  "reference" discharge cycles retained (decimated ×2). 158–163 samples. Noisier.

**Target:** `capacity_retention` = capacity_ahr / baseline_ahr ∈ [0, 1]. SOH label.
Derived from discharge integration → input is from charge → no leakage.

**Validation:** Leave-One-Battery-Out (LOBO), 13 folds. Always hold out entire batteries.

**Input tensor:** 128 resampled points per charge cycle. Currently channels = [V, |I|, T].
Segment = first V≥3.8V to last |I|≥0.04A (CC + full CV taper).

---

## Baselines to Beat

| Model | Task | MAE | R² |
|---|---|---|---|
| RF (scalars) | SOH micro | **0.041** | 0.67 |
| RF (r_proxy) | RUL practical | 0.032 | 0.966 |
| TCN macro-clock | RUL practical | 0.042 | 0.935 |

The micro RF uses 9 hand-crafted scalars from the same charge window (V_mean, I_mean, I_std,
T_mean, T_max, T_rise, I_droop, V_mid, T_std). It improves with CC+CV data (was 0.048 on
CC-only). This is the direct target.

---

## What We Built

Three architectures, all receiving (128, 3) tensors:

**CNNReg** — 3 conv blocks (32→64→64 ch, GroupNorm, kernel 5/5/3), global concat_pool.
~24k params.

**TCNRegPooled** — 3 dilated non-causal blocks (32ch, dilations 1/2/4, GroupNorm, kernel 5),
global concat_pool. ~11k params.

**LSTMReg** — single-layer LSTM (hidden=64), last hidden state → dropout → FC. ~18k params.

**concat_pool:** `cat[mean, max, last]` over the 128 time steps. 3× richer than mean-only;
designed to capture the monotone-ramp endpoint.

**Hybrid variants:** same backbones but the 9 RF scalars are concatenated with the pooled
embedding before the FC head. Theoretically ≥ RF (has all RF info + curve shape).

**Training:** cosine LR schedule, per-batch jitter (σ=0.005), LR=3e-4 all models, patience=15,
batch=256. Controlled inner-val battery (B0018). 8 parallel workers.

---

## Current LOBO Results (latest run on CC+CV with |I| fix)

| Model | MAE ± std | R² | Skill |
|---|---|---|---|
| RF (scalars) | **0.041** ± 0.044 | 0.67 | 0.68 |
| CNN | 0.16 ± 0.08 | −1.4 | −0.21 |
| TCN | 0.25 ± 0.12 | −3.6 | −0.71 |
| LSTM | ~0.10 (in progress) | TBD | TBD |
| Hybrid-TCN | 0.26 ± 0.08 | −5.0 | −1.03 |

DL is still not beating RF. Best is LSTM at ~0.10. **RF is 2× better than our best DL.**

---

## What We Already Fixed

1. **Pooling:** `mean` → `concat_pool[mean, max, last]` — fixed CNN/TCN negative R²
2. **Normalization:** BatchNorm → GroupNorm — fixed LOBO domain-shift collapse
3. **LR:** all models 1e-3 → 3e-4 — fixed CNN overshooting (was stopping at epoch 2)
4. **Training:** cosine LR, per-batch jitter, controlled inner-val battery
5. **CC+CV extraction:** extended from CC-only (3.8–4.1V) to CC+full CV (3.8V→I≥0.04A)
   — RW tensors went from 1–13 raw pts to 188–1862 pts
6. **Sign fix:** raw I → |I| — controlled (+1.5A) and RW (−2A) now same direction
7. **Threading:** init lock + per-fold Generator — fixed RNG race across 8 workers
8. **Reproducibility:** deleted stale-CSV reload cell, fixed RUL blowup (1e14)

---

## Why DL Is Still Losing — Current Hypotheses

### H1: 794 samples is just not enough for these architectures
Macro-clock TCN (notebook 04) also lost to RF on 13 trajectories. The micro-clock gives
us 794 samples which is better, but: with 13 LOBO folds each using ~11 batteries for training
(~580 samples), each fold has very little data. The controlled cohort (636 samples) is
clean but 4 battery *types*; the RW cohort (158) has domain diversity but only 6 cycles
average per battery fold.

RF works because it uses 9 hand-crafted features that summarise the relevant physics.
DL needs to learn these features from scratch in each fold.

### H2: The `last` component of concat_pool is constant by construction
The segment ends when |I| = 0.04A (by the i_cut criterion). So:
- `last(V)` ≈ 4.2V — constant for all samples
- `last(|I|)` ≈ 0.04A — constant for all samples  
- `last(T)` — varies, informative

Two-thirds of the `last` vector is constant noise. This wastes head capacity and may
actually hurt by pushing the optimizer toward the T-only last signal.

**Fix candidate:** change concat_pool to `cat[mean, max]` only, or to `cat[mean, max, first]`
(first timestep is where V/I variation across cycles is meaningful).

### H3: Temporal structure is still partially misaligned between cohorts
Even with |I| and V_LO=3.8:
- Controlled: points 0–30 = CC ramp (V rising, |I| constant ~1.5A), points 30–127 = CV
  (V flat, |I| decaying)
- RW: points 0–5 = brief CC (or immediate CV), points 5–127 = CV decay (V flat, |I| ≈ 2A → 0)

LSTM processes left-to-right so position matters. The "start of CV" is at position ~30 for
controlled and ~5 for RW — model has to learn this implicitly. CNN/TCN global pool is less
sensitive to this.

### H4: The charge scalars already capture most of the information
The 9 RF scalars (I_mean, I_std, I_droop, T_rise, etc.) are highly compressed summaries of
the curve that turn out to be near-sufficient for SOH prediction. The raw 128-point curve
may not add much beyond what these 9 numbers already encode at this data scale.

Evidence: even our Hybrid-TCN (which gets both the scalars AND the curve) performs worse
than RF alone. This suggests the DL backbone is *hurting* performance rather than helping.

### H5: GroupNorm + tiny batches = unstable normalization
With batch=256 but only ~580 train samples per fold → ~2–3 batches per epoch. GroupNorm
computed on 2–3 batches of tiny controlled-battery data may be noisy.

---

## Options to Explore

### Option A: Fix concat_pool (quick, no data changes)
Drop `last` (constant by construction) and use `cat[mean, max]` only, or `cat[mean, max, first]`.
`first(V)` varies meaningfully with battery age (the starting voltage of the CC segment
reflects open-circuit voltage = state of charge = health proxy).
*Effort: 1 line in curve_models.py + rebuild*

### Option B: Add the Q(t) channel (Stage 4.5)
Normalized Q(t)/Q_max as a 4th channel — the area under the I-decay curve directly encodes
how much charge the cell accepted in CV, which is a strong capacity proxy. Using Q/Q_max
avoids the leakage issue (we normalise by the cycle's own maximum, not absolute capacity).
*Effort: update slice_resample, rebuild .npz, update ARCH_MAP with C=4*

### Option C: Attention pooling instead of concat_pool
Replace global max/mean/last with a learned attention weight over the 128 timesteps.
The model learns WHICH timesteps are most informative per architecture.
`attn = softmax(W @ x)` → weighted sum. Adds ~128 params.
*Effort: new Attention1D class in curve_models.py*

### Option D: Align cohorts by resampling in current-space, not time-space
Instead of 128 evenly-spaced TIME points, resample 128 evenly-spaced CURRENT points
(from |I|_max down to |I|_cut). Both cohorts then produce:
- Position 0: V/T at peak current
- Position 127: V/T at near-zero current
This makes the 128 positions mean the same thing across controlled and RW.
*Effort: change slice_resample interpolation axis; same downstream code*

### Option E: Forget trying to beat RF with pure DL — lean into Hybrid
Our theory says Hybrid ≥ RF. In practice Hybrid is worse. The issue may be that:
a) The DL backbone introduces noise that hurts the scalar signal, or
b) The hybrid model needs more data to outperform the RF-only path.

**Sub-option E1:** Use the RF's 9 scalars as inputs to a simple MLP (no curve) — just to
confirm the scalars are extractable via DL before adding curve complexity. If MLP-on-scalars
≈ RF, the baseline is fair.

**Sub-option E2:** Try Hybrid with a frozen or heavily regularized backbone (weight_decay=1e-3
or dropout=0.5) so the DL part is forced to add value marginally rather than overwriting
the scalar signal.

### Option F: Move to a simpler 1D curve feature
dQ/dV incremental capacity analysis (IC): plot differential capacity vs voltage. The IC
curve has distinctive peaks that shift with degradation. Computing dQ/dV and using that as
the 128-pt input (V axis instead of T axis) is the state-of-art charge-curve SOH method.
*Effort: new channel derived in slice_resample; no model changes*

### Option G: Accept that DL doesn't win on the micro-clock either
Narrative pivot: "We built the micro-clock DL pipeline, it achieves competitive performance
with the RF on controlled batteries (LSTM ~0.03 on Tier-1), but generalisation to
randomized profiles degrades. This is consistent with the macro-clock finding: 13-battery
LOBO is fundamentally underdetermined for DL. The macro RF remains the best deployed model."

Present LSTM as "best DL", show it competes on controlled folds, explain the RW
generalisation gap honestly. The presentation story becomes: RF for deployment, DL for
understanding curve-shape effects.

---

## Most Likely Root Cause (my read)

The data has a structural problem: **controlled batteries dominate (636/794 = 80%)** and
have very different curve shapes from RW. Any LOBO fold that holds out an RW battery is
essentially asking the model to extrapolate from controlled→RW with minimal RW training
data. RF generalises because its 9 scalars are invariant to the cohort-level structural
differences. DL has to learn cohort-invariant features from scratch in each fold.

Highest leverage options: **A** (fix the dead `last` component), **D** (current-space
resampling for alignment), **B** (Q channel, but be careful about the leakage narrative).

Option G is always available as a fallback and is actually a solid academic story.

---

## Numbers Reference

| Metric | Value | Context |
|---|---|---|
| RF SOH LOBO MAE | 0.041 | Charge-curve scalars, CC+CV data |
| RF RUL LOBO MAE | 0.032 | r_proxy scalars, practical phase |
| LSTM Tier-1 val loss | 0.002 | MSE on B0018, not generalisation |
| LSTM LOBO MAE (current) | ~0.10 | In progress |
| Macro TCN RUL MAE | 0.042 | Notebook 04, macro clock |
| Samples total | 794 | After CC+CV fix |
| Controlled samples | 636 | B0005–B0007, B0018 |
| RW samples | 158 | 9 batteries, ref cycles only |
| Batteries (LOBO folds) | 13 | One held out per fold |
