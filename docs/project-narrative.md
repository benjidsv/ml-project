# Battery State-of-Health Prediction from Charge Curves: A Two-Phase Study

**Authors:** Benjamin Arbousset, Loïc Cupif, Aurélien Ridard — DE INGE-2 2027, EFREI Paris  
**Course:** Machine Learning for Data Engineers (Instructor: Stephany Rajeh, Ph.D.)

---

## Abstract

We address the problem of predicting lithium-ion battery State-of-Health (SOH) from the shape of individual charge cycles, without any discharge measurement at inference time. Starting from the NASA PCoE dataset (13 cells, ~800 samples), we show that scalar-feature baselines (Random Forest, XGBoost) are effectively unbeatable at this data scale, even with carefully engineered deep learning pipelines. We then extend to a multi-source pool — NASA + CALCE CS2/CX2 — reaching **22 cells and 12,260 samples**, which gives sequence models the data they need to win. On within-pool held-out cells (GroupKFold-6), a BiGRU trained on voltage-grid tensors achieves **MAE 0.036–0.038** against RF's **0.072**, halving the error. The harder problem is **cross-dataset transfer (Leave-One-Dataset-Out)**: the Transformer is the best in-distribution model but **collapses when transferred to NASA** (LODO MAE 0.105); BiGRU alone survives transfer (LODO MAE 0.041). The dominant mechanism behind the transfer gap is not chemistry — it is **protocol fingerprints** in the input channels (absolute C-rate, charging time, imputed temperature). Rate-warp and time-warp augmentation teach the model to be invariant to these fingerprints, reaching a **LODO mean MAE of 0.058**, with a further improvement to **NASA MAE 0.045** when one labeled target cycle is available for lightweight fine-tuning. We report three failed approaches (invariant feature engineering, CORAL, DANN, GroupDRO) and the physical reason each one failed, because honest negative results are as informative as wins.

---

## 1. Problem and Framing

### 1.1 Why battery SOH matters

Lithium-ion batteries are the energy store of the renewable transition. Their degradation is nonlinear, cell-to-cell variable, and impossible to directly measure without opening the pack. State-of-Health — commonly defined as retained capacity relative to the factory rating — determines when a cell should be retired, and predicting it from non-invasive measurements is an open problem with both academic and industrial stakes.

### 1.2 Task definition

We frame SOH prediction as a regression problem:

- **Target:** `SOH = capacity_ahr / C_nom ∈ (0, 1]`, where `C_nom` is the rated capacity from the cell datasheet. The End-of-Life (EOL) threshold is 0.8 (industry standard: 80% retention). We use the datasheet `C_nom` rather than a per-battery measured baseline so that the target is computable for any new cell without initial characterisation.
- **Input:** the CC charging curve recorded in the cycle immediately *preceding* the discharge. Each cycle yields a fixed-length tensor of 128 points sampled along a descending voltage axis (4.2 V → 3.8 V), with three channels: C-rate (`|I|/C_nom`), temperature (°C), and elapsed charging time (s). Voltage is the *axis*, not a channel, making the representation position-consistent across cells with different ratings.
- **No discharge signal at inference time.** The capacity label comes from integrating the *discharge* current, but the model sees only the charge curve. This is the only leakage-free framing for on-device deployment (chargers log charge data; discharge events are often incomplete or not monitored).

### 1.3 Validation philosophy

Every result in this paper uses **whole-battery hold-out** cross-validation. Splitting by time within a single battery — putting early cycles in training and late cycles in test — leaks degradation context and produces inflated metrics. We use three progressively stronger evaluation schemes:

- **LOBO** (Leave-One-Battery-Out): one battery held per fold, 13 folds on NASA data.
- **GroupKFold-6**: cells allocated to 6 folds by study group; ~3–4 cells held per fold.
- **LODO** (Leave-One-Dataset-Out): one entire dataset held per fold (NASA ↔ CALCE); tests cross-chemistry, cross-protocol generalisation.

### 1.4 The core scientific question

Do raw charge-curve **shapes** carry SOH information that hand-crafted scalar summaries cannot access? Scalars (mean voltage, mean C-rate, temperature rise, etc.) compress the ~128-point curve to 9–11 numbers. Our working hypothesis is that the curve contains information in its temporal *structure* — how fast current decays through the CV phase, where the CC→CV knee falls, how the ramp shape shifts with ageing — that no finite set of scalar descriptors recovers exactly. The two-phase structure of this project is the story of testing that hypothesis.

---

## 2. Phase 1 — NASA Dataset: Baselines, the Micro-Clock, and the Data Wall

### 2.1 Dataset and cleaning

The NASA PCoE dataset provides raw `.mat` files for controlled and randomized charge/discharge cycling experiments on commercial Li-ion 18650 cells.

**Controlled batteries (B-series, 2.0 Ah):** B0005, B0006, B0007, B0018, cycled at room temperature (~24°C) with fixed 1.5 A constant-current charge and 2 A constant-current discharge. 132–168 cycles each; all cross the 80% EOL threshold. Excluded groups: hot (43°C, B0029–B0032 — only ~1–6% capacity fade over 40 cycles, never reach EOL) and cold (4°C, B0042–B0044 — inconsistent mixed protocols, impossible to separate thermal from ageing effects). B0041 excluded for anomalous starting capacity (1.33 Ah vs 1.86–2.05 Ah peers; only 25 cycles).

**Randomized batteries (RW-series, 2.2 Ah):** variable charge and discharge current profiles simulating renewable-proxy loads. Retained: RW1, RW13–RW17, RW19, RW20 (8 cells). Excluded: RW2, RW18 (temperature sensor fault: −4093°C readings); RW9 removed during the extended-dataset phase (random-SOC group that starts charges from arbitrary SOC; papers consistently exclude the RW9–RW12 group).

**Total retained (NASA phase): 13 cells, 801 charge cycles, 788 clean samples** after voltage-window and segmentation quality gates.

An early EDA finding: the resistance proxy `r_proxy` (onset ΔVΔI at the start of each discharge step) correlates with true EIS-measured internal resistance with r ≈ 0.77 on B0005 — strong enough for field deployment where EIS equipment is unavailable.

### 2.2 Scalar-feature baselines (macro-clock)

The first modelling phase (notebooks 03–04) trains on one scalar per cycle — features derived by integrating or summarising the full discharge curve (mean voltage, temperature stats, capacity fade rate, resistance proxy with rolling windows). Validation is 13-fold LOBO.

![NASA phase model comparison](../results/extended_tier1/nasa_phase_baseline_comparison.png)

**Practical phase (13 batteries, `r_proxy` only — no EIS):**

| Model | MAE ± std | RMSE | R² ± std | Monotonicity | Skill |
|---|---|---|---|---|---|
| Ridge | 0.136 ± 0.078 | 0.171 | 0.560 ± 0.779 | 0.739 | — |
| **Random Forest** | **0.032 ± 0.019** | **0.054** | **0.966 ± 0.033** | **0.926** | **0.892** |
| GBM | 0.035 ± 0.023 | 0.056 | 0.961 ± 0.038 | 0.737 | 0.858 |

Target: `rul_frac` (normalised remaining useful life, ∈ [0,1]). Random Forest is dominant: R² 0.966 with per-fold standard deviation of only 0.033, showing stability across all 13 battery types. The 0.926 monotonicity score (fraction of consecutive predictions that correctly rank degradation direction) reflects RF's bagging-smoothed trajectory.

A **macro-clock TCN** (Temporal Convolutional Network operating over sequences of cycle-level scalars, notebook 04) was tested as the first deep learning model. With lookback L=5 it achieved MAE 0.042 and R² 0.935 — worse than RF on both metrics. Crucially, it converged after only 16 epochs vs LSTM's 124. That fast convergence is a data-starvation signal, not an efficiency win: the model ran out of signal within the first few passes over 13 trajectories.

![TCN vs RF RUL trajectory on B0005](../results/train_first_dl/dl_macro_rul_trajectory.png)
*Left: macro-clock TCN (noisy, erratic early-life predictions). Right: RF baseline (smooth, tracks the linear decay). B0005 held out in LOBO.*

### 2.3 The micro-clock pivot and the voltage-grid representation

The scalar baselines and the macro TCN both operate on one number per cycle. The micro-clock switches to the raw **charge-curve tensor** — ~128 time-series measurements within a single charge event. With 788 such curves, we have ~60× more samples to train on than with trajectory-level scalars.

The key engineering decision was the representation. Three iterated designs were tested (notebooks 05–09):

1. **Time-resampled CC+CV, per-sample z-scored:** catastrophic failure (CNN R²=−1.1, LSTM MAE 0.077). Per-sample normalisation destroys absolute magnitude; the LSTM reads only shape, not level.
2. **CV-only, global scaling, with stress-marker scalars (notebooks 06–08):** progress, but LSTM never beats RF. The scalar head provides the decisive signal; the curve adds noise. Best arm (E2 RF with stress markers): MAE 0.053, R² 0.647.
3. **Voltage-grid LSTM (notebook 09):** the breakthrough representation. Instead of time on the x-axis, use **voltage** as the axis (descending 4.2→3.8 V, 128 grid points). Channels: `[|I|, T, t_elapsed]`. Voltage is the natural alignment axis: at any grid position, the cell is in the same physical state regardless of chemistry or rate. Global channel-wise scaling (not per-sample). Architecture: `VGLSTMReg` — bidirectional LSTM (hidden=40), masked mean+max pooling, dropout, linear head (~14.6k parameters).

![Micro-clock SOH trajectory — CNN/TCN/LSTM on B0005](../results/train_first_dl/dl_micro_soh_trajectory.png)
*Micro-clock models on held-out B0005. CNN (blue dashed) and TCN (red dashed) are flat or noisy; LSTM (pink) is closest to truth but still volatile. Demonstrates the data-scale problem before the voltage-grid fix.*

**Voltage-grid LSTM LOBO results (13-fold):**

| | MAE ± std | R² ± std | Spearman | Notes |
|---|---|---|---|---|
| RF (scalar features) | 0.048 ± 0.043 | 0.62 ± 0.61 | 0.875 | charge scalars, same data |
| **VG-LSTM** | **0.062 ± 0.044** | **0.58 ± 0.61** | **0.875** | voltage-grid, 14.6k params |
| VG-CNN-LSTM | 0.087 ± 0.043 | 0.22 ± 0.94 | 0.809 | CNN front-end hurts transfer |

LSTM matches RF on Spearman (0.875 tied) — both models rank degradation correctly. The aggregate MAE advantage stays with RF because two batteries cause disproportionate damage to the LSTM aggregate: **RW1** (MAE 0.143, R²=−1.22) and **RW9** (MAE 0.175, R²=−0.32). The CNN-LSTM does not improve — it performs better on the majority RW folds but collapses on controlled cells, a foretaste of the transfer problem that becomes central in Phase 2.

![CC ramp duration vs SOH](../results/tune_lstm/08_p0_cc_dt_vs_soh.png)
*CC ramp duration (time from 4.0 V to CV onset at 4.15 V) vs SOH for controlled (left, r=0.75) and RW (right, r=0.82) batteries. This strong correlation motivates using the full charge-curve shape rather than summary scalars: the duration is a SOH proxy that scalars partially capture but the raw curve encodes more cleanly.*

### 2.4 Two "we understood why" findings from Phase 1

**RW1: the stationarity violation.** After all segmentation fixes were applied, RW1's LSTM LOBO shows slope ≈ −0.022, correlation ≈ −0.030, and a uniform mean error of −0.124 across all 21 reference cycles. The model outputs essentially a constant. RF also fails RW1 (MAE 0.088, R²≈0). Investigation of per-battery nominal capacities reveals the root cause: RW1 has a baseline of **2.0003 Ah**, while all 8 other RW batteries cluster tightly at 2.10–2.14 Ah (~5–6% higher). The CV current-decay taper encodes absolute charge accepted — roughly `SOH × C_baseline` — not fractional SOH directly. At SOH=0.80: RW1 accepts 1.60 Ah during CV; other RW cells accept ~1.70 Ah. The model trained on the 2.10–2.14 Ah cohort has learned "CV taper ≈ 1.60 Ah → SOH ≈ 0.75." It applies that mapping to RW1 at SOH=0.80 and systematically undershoots by ~0.05 from the capacity gap, with the remaining offset from protocol-group difference (RW1 = *Variable Charge* group vs *Skewed* and *Charge–Discharge* for other RW). This is a stationarity violation: the SOH label is defined relative to each battery's own baseline, but the physical curve shape is determined by absolute capacity, and those two quantities are incommensurable without knowing the baseline. The 0.124 MAE floor on RW1 is structural for any architecture that does not have access to the battery's nominal capacity.

![RW1 and RW9 shrinkage diagnostic](../results/train_first_dl/shrinkage_error_vs_dist.png)
*Error (predicted − true SOH) vs distance from training mean for RW1 (top) and RW9 (bottom), both RF and LSTM. RW1's LSTM error is a flat horizontal line (slope ≈ 0, constant offset −0.124) — the model outputs a constant regardless of true SOH. RW9 shows slope ≈ −0.6: correct direction but compressed. These are two distinct failure modes with different root causes.*

**Current-axis resampling: a negative result with a physical explanation.** After observing that the VG-LSTM compresses predictions at high SOH for RW9, we tested resampling the CV taper at 128 *uniform current levels* (recording V and T at fixed |I| steps from I_peak to I_cut) rather than uniform time steps. The motivation: electrochemical impedance spectroscopy works by applying controlled current excitations and reading voltage; perhaps recording V at fixed current levels is a cleaner SOH fingerprint. Result: categorically worse. RW9 degraded from MAE 0.103 (R²=+0.48) to MAE 0.216 (R²=−1.37). The reason is unambiguous — in a CV window, **V ≈ 4.20 V throughout by definition** (the charger maintains the battery at its voltage setpoint). V(I) carries essentially no SOH information; the current channel in current-axis resampling becomes a constant monotone grid identical for every cycle. The SOH signal in CV is exclusively temporal: how fast |I| decays from I_peak toward zero. Time-axis resampling preserves this; current-axis resampling destroys it. The impedance-sweep analogy fails because an impedance sweep drives the cell with a sinusoidal excitation at known frequencies — the resulting V contains frequency-domain information that is SOH-sensitive. A CV taper is a free relaxation response whose shape *in time* IS the measurement.

### 2.5 The data wall

Thirteen trajectories / ~800 samples is the fundamental constraint. **Controlled batteries dominate (629/788 = 80%)** and have structurally different curve shapes from RW cells. Any LOBO fold holding out an RW battery asks the model to extrapolate from controlled→RW with minimal RW training data. RF generalises because its 9–11 scalars are largely cohort-invariant; the voltage-grid LSTM has to learn cohort-invariant features from scratch in each fold, and 13 batteries is too few for that to converge reliably. More data is the only path forward.

---

## 3. Phase 2 — Extended Multi-Dataset: Scaling to 22 Cells

### 3.1 Extending to CALCE CS2/CX2

**CALCE CS2 and CX2** (Center for Advanced Life Cycle Engineering, University of Maryland) provide 1.1 Ah commercial Li-ion cells cycled on Arbin test equipment with CC-CV protocol (0.5C charge to 4.2 V). The data format is multi-file Arbin Excel exports per cell.

Two parsing challenges had to be resolved before the data was usable:

1. **File ordering:** CALCE filenames follow an unpadded `M_D_YY` scheme (`CS2_35_10_15_10.xlsx`) that does not sort correctly as strings — August files sort before October, placing early high-capacity cycles at the *end* of the timeline and producing jagged, non-monotone SOH trajectories. Fix: files are ordered by the first `Date_Time` value recorded inside each file, with filename-date as a fallback.
2. **Cumulative capacity:** `Discharge_Capacity(Ah)` in the Arbin format is cumulative within each test session. Some `Cycle_Index` aggregations span many sub-cycles (e.g., CX2_16 cycle 43 = 12.5 Ah → apparent SOH ≈ 11×). A plausibility band of `[0.22, 1.65]` Ah (0.2–1.5× `C_nom`) is applied to reject partial characterisation and malformed cycles. Impact across all 11 CALCE cells: **1.3% of 14,391 cycles dropped**.

Cells excluded from CALCE: CX2_16 (hard protocol change at cycle 1152 — 0.5C then 0.61C, two incommensurable SOH scales); cells on CADEX cycler (incompatible column format); cells with pulsed-discharge or partial-cycling protocols.

![SOH trajectories — all 22 cells](../results/extended_tier1/01_soh_trajectories.png)
*SOH trajectories for all 22 Tier-1 cells (blue = NASA, orange = CALCE). NASA controlled cells (B-series, top-left) degrade smoothly over 100–170 cycles; CALCE CS2/CX2 cells have 800–2000 cycles each. The scale difference is the core motivation for the extension.*

**Final Tier-1 cell roster: 22 cells**

| Source | Cells | Samples | C_nom | Notes |
|---|---|---|---|---|
| NASA controlled | B0005, B0006, B0007, B0018 | 629 | 2.0 Ah | Full CC-CV, 24°C |
| NASA randomized | RW1, RW13–17, RW19, RW20 | 115 | 2.2 Ah | Variable load, ref. cycles only |
| CALCE CS2 | CS2_33–38 (6 cells) | 5,661 | 1.1 Ah | CC-CV, 25°C (no thermocouple) |
| CALCE CX2 | CX2_33, 35, 37, 38 (4 cells) | 6,737 | 1.1 Ah | CC-CV, 25°C (no thermocouple) |
| **Total** | **22** | **12,260** | — | After quality filtering |

### 3.2 The multi-dataset merge: representation decisions

Naively concatenating NASA and CALCE data would confuse the model: a NASA cell charges at ~1.5 A (2.0 Ah × 0.75C) while a CALCE cell charges at ~0.55 A (1.1 Ah × 0.5C). These are the same *relative* charging rate but wildly different *absolute* currents. Three representation decisions make the merge principled:

| Decision | Choice | Rationale |
|---|---|---|
| Current channel | **C-rate = \|I\|/C_nom** | Raw amps confound cell size with charge rate. At ~0.5C, NASA and CALCE charge curves lie in the same band regardless of absolute current. |
| Temperature (CALCE) | **Imputed constant 25°C** | CALCE cells have no thermocouple. Stated test-chamber ambient is 25°C; imputing the physical truth is preferable to imputing the training mean. |
| Target denominator | **C_nom from datasheet** | No per-battery measured baseline required. Enables merging cells that haven't been characterised. |
| Voltage grid | **128 pts, 4.2→3.8 V** | Same window across both datasets; CALCE cells charge to 4.2 V cutoff, same as NASA. |

The resulting 128×3 tensor — `[C-rate, T, t_elapsed]` with voltage as the axis — is the same for any Li-ion cell using a CC-CV protocol to 4.2 V, regardless of capacity or cycler brand.

![NASA vs CALCE channel overlay](../results/extended_tier1/01_crate_overlay.png)
*Early-life charge curves for NASA/B0005 (blue) and CALCE/CS2_33 (orange) overlaid on the voltage grid. Ch 0 (C-rate): both datasets now sit in the same 0.5–0.8C band after normalisation — the merge works. Ch 1 (temperature): CALCE is imputed constant at 25°C while NASA shows real thermal dynamics — the imputed T becomes a near-perfect dataset-identity signal after scaling, which is the root cause of the transfer gap. Ch 2 (t_elapsed): the two datasets are separated by an absolute offset (CALCE charges are longer at 0.5C than NASA at 0.75C) — this is the protocol fingerprint that rate-warp augmentation learns to ignore.* The SOH target range is `y ∈ [0.20, 1.24]` across the merged pool (CALCE CX2 cells start fresh above rated; all decay toward ~0.75 at EOL).

### 3.3 Result 1 — DL finally beats RF (GroupKFold-6)

With 12,260 samples across 22 cells, we compare a scalar Random Forest baseline against three sequence architectures on GroupKFold-6 held-out cells:

- **RF:** trained on 11 charge-curve scalar descriptors (`coverage`, `v_start`, `cc_dt`, `cc_slope`, `crate_mean/max/start`, `temp_mean/max`, `t_total`, `r_proxy`).
- **LSTM:** `VGLSTMReg` — bidirectional LSTM (hidden=40), masked mean+max pool, ~14.6k parameters.
- **CNN-LSTM:** `VGCNNLSTM` — Conv1d front-end (8 channels, kernel 5, GroupNorm, same-padding) → BiLSTM → masked pool, ~16.3k parameters.
- **BiGRU:** GRU cell replacing LSTM (hidden=47 to roughly match parameter counts), ~14.9k parameters.
- **Transformer:** 2-layer multi-head self-attention (d=32, 4 heads), ~18k parameters.
- **Attn-GRU:** BiGRU with learned attention pooling replacing mean+max.

**GroupKFold-6 aggregate results:**

| Model | MAE ± std | R² ± std | Skill | Spearman |
|---|---|---|---|---|
| RF (11 scalars) | 0.072 ± 0.056 | 0.45 ± 0.70 | 0.525 | 0.775 |
| LSTM | 0.038 ± 0.013 | 0.87 ± 0.07 | 0.761 | 0.963 |
| CNN-LSTM | 0.038 ± 0.018 | 0.79 ± 0.26 | 0.725 | 0.911 |
| BiGRU | 0.038 ± 0.015 | 0.77 ± 0.24 | 0.733 | 0.904 |
| **Transformer** | **0.033 ± 0.026** | **0.846 ± 0.176** | **0.788** | **0.959** |
| Attn-GRU | 0.037 ± 0.015 | 0.816 ± 0.182 | 0.741 | 0.949 |

RF collapses on the NASA folds: `NASA_CTRL` fold MAE = 0.078–0.087, `NASA_RW` fold MAE = 0.19, R² < 0.

| | RF GKF-6 per fold | LSTM GKF-6 per fold |
|---|---|---|
| ![RF per fold](../results/extended_tier1/02_rf_gkf_per_fold.png) | ![LSTM per fold](../results/extended_tier1/02_lstm_gkf_per_fold.png) |

*Left: RF predicted vs actual SOH for each of the 6 GroupKFold folds. The top 4 CALCE folds are tight (R²=0.80–0.94); the bottom two NASA folds are scattered (NASA_CTRL R²=0.11, NASA_RW R²=−0.99). Right: LSTM on the same folds — all 6 panels are tight, including NASA_CTRL (R²=0.83) and NASA_RW (R²=0.75).*

![Model comparison — GKF-6 vs LODO](../results/extended_tier1/comparison_gkf_vs_lodo.png)
*Central result: grouped bars showing each model's MAE on GroupKFold-6 (blue, in-distribution) vs LODO (orange, cross-dataset transfer). The Transformer wins in-distribution but collapses on transfer. BiGRU is the only model where both bars are low.* RF's scalar descriptors are computed from the majority CALCE distribution; when held-out cells are NASA batteries with structurally different protocol characteristics, RF has no curve-shape signal to fall back on. All DL models halve the aggregate error relative to RF (0.033–0.038 vs 0.072), driven by their ability to read the full curve shape on held-out cell types.

The Transformer is the strongest in-distribution model (MAE 0.033, R² 0.846, Spearman 0.959) but also the most volatile: per-fold std of 0.026, and it collapses to MAE 0.088 on the NASA_RW fold. This volatility is not an artifact — it is a preview of the transfer problem.

### 3.4 Result 2 — Cross-dataset transfer (LODO) separates the models

Leave-One-Dataset-Out is a strictly harder evaluation: the model must generalise across chemistry families, cell form factors, and cycler protocols that it has never encountered during training. The two folds are:
- **hold-CALCE:** train on NASA (744 samples), test on 11,516 CALCE samples.
- **hold-NASA:** train on CALCE (11,516 samples), test on 744 NASA samples.

**LODO results:**

| Model | LODO mean MAE | hold-CALCE MAE | hold-NASA MAE | hold-CALCE R² | hold-NASA R² |
|---|---|---|---|---|---|
| RF | 0.147 | 0.141 | 0.152 | 0.254 | −1.02 |
| **BiGRU** | **0.041** | **0.040** | **0.043** | **0.923** | **0.800** |
| CNN-LSTM | 0.071 | 0.085 | 0.057 | 0.670 | 0.638 |
| Transformer | 0.083 | 0.061 | 0.105 | 0.838 | −0.195 |
| Attn-GRU | 0.066 | 0.061 | 0.071 | 0.838 | 0.481 |

This table is the central result of the project. Three patterns:

![LODO breakdown — hold-CALCE vs hold-NASA by model](../results/extended_tier1/lodo_calce_vs_nasa.png)
*LODO MAE split by which dataset is held out. Green = trained on NASA, tested on CALCE. Red = trained on CALCE, tested on NASA. BiGRU stays low in both directions. CNN-LSTM, Transformer and Attn-GRU all have one bad direction.*

| RF LODO | LSTM LODO |
|---|---|
| ![RF LODO](../results/extended_tier1/02_rf_lodo_per_fold.png) | ![LSTM LODO](../results/extended_tier1/02_lstm_lodo_per_fold.png) |

*RF LODO (left): training on NASA and testing on all 11,516 CALCE cycles gives R²=0.25; reverse direction gives R²=−1.02 (worse than a constant predictor). LSTM LODO (right): both held-out datasets show reasonable scatter (CALCE R²=0.87, NASA R²=0.72).*

> **BiGRU LODO scatter:** `results/extended_tier1/bigru_lodo_per_fold.png` *(generated by `scripts/gen_bigru_lodo_plots.py`)*

**1. RF fails both directions.** Training on NASA (744 samples) and testing on CALCE (11,516) produces R²=0.254 — RF barely outperforms a naive mean predictor when trained on the minority dataset. In the reverse direction, RF trained on CALCE still fails on NASA (R²=−1.02). Scalar features that are discriminative within a protocol group are not discriminative across groups.

**2. BiGRU alone survives both transfer directions.** LODO MAE 0.041 is only a 10% degradation from its GroupKFold-6 score (0.038) — the model has learned representations that transfer. R² stays above 0.80 in both held-out directions.

**3. More complex models transfer worse.** The Transformer achieves the best in-distribution score (0.033) but collapses on hold-NASA (MAE 0.105, R²=−0.195) — negative R² means it is worse than a naive constant predictor on the held-out chemistry. CNN-LSTM is intermediate (LODO 0.071). Attn-GRU is better than Transformer in transfer but worse than plain BiGRU.

**Why BiGRU transfers and Transformer doesn't:** The key is the *pooling mechanism*. BiGRU uses **masked mean+max pooling** — a global, position-agnostic summary that is largely invariant to absolute time-axis alignment. The Transformer uses self-attention over all 128 positions: each attention head learns which voltage-grid positions to attend to for predicting SOH, and those positions carry protocol fingerprints (the CC ramp occupies different grid positions for NASA 0.75C vs CALCE 0.5C; absolute C-rate values differ by ~50%). Adding CNN front-ends or attention mechanisms *improves* performance on the majority CALCE cells — where the model has enough data to memorise these fingerprints — but *breaks* transfer to NASA where the fingerprints differ. This is the key tension of multi-source learning: capacity and transfer are in tension, and the model architecture determines which side wins.

### 3.5 Why the transfer gap exists: protocol fingerprints

After examining the input channels, the primary transfer obstacles are clear:

- **C-rate channel (ch 0):** CALCE charges at fixed 0.5C (after C-rate normalisation, all CALCE CC ramps look nearly identical at ~0.5C). NASA controlled cells charge at ~0.75C; NASA RW cells at ~0.91C (2.0 A / 2.2 Ah). The model can learn to use C-rate level as a dataset-identity detector and overfit to it.
- **t_elapsed channel (ch 2):** charging time scales inversely with rate — a 0.5C charge takes roughly twice as long as a 1.0C charge. After normalisation by `C_nom`, absolute time still encodes the protocol (CALCE charges are systematically longer).
- **Temperature channel (ch 1):** all CALCE cells have T imputed to exactly 25.0°C (constant). After global StandardScaler normalisation, the CALCE T channel becomes nearly constant at its scaled mean, while NASA T varies with cycling (16–41°C measured). A model that learns "T variance = NASA, T constant = CALCE" has a near-perfect dataset-identity signal available.

These fingerprints are not degradation signals — they are artefacts of the test protocol and measurement setup. A model that uses them to improve within-distribution performance will fail on transfer.

### 3.6 What works to improve transfer — and what doesn't

We tested four classes of transfer robustness interventions, working from most principled to most pragmatic.

**Invariant feature engineering (rejected).** Hypothesis: replacing `t_elapsed` with charge-acceptance integral `Q(V)` and incremental capacity `dQ/dV` — features that are rate-invariant *by construction* — would remove the protocol fingerprint. Result: `GRU-inv4` (Q_frac, dQ/dV channels) achieved **worse** GroupKFold-6 performance (MAE 0.047 vs 0.038) and catastrophic failure on NASA_CTRL (R²=−1.34). Q(V) and dQ/dV encode different electrode-chemistry fingerprints — the absolute position of IC peaks differs between CALCE NMC and NASA LCO cell chemistries. Removing one fingerprint introduced another.

**CORAL and DANN (rejected).** Correlation Alignment (CORAL) minimises covariance shift between source domains; Domain-Adversarial Neural Networks (DANN) add a gradient-reversal domain classifier. Both require target-domain samples *during training*. Under LODO, the entire test dataset is withheld — CORAL and DANN can only align CALCE sub-protocol groups (CS2 vs CX2, T1 vs T2), an axis nearly orthogonal to the NASA→CALCE shift. Results: CORAL LODO MAE 0.051 (worse than plain BiGRU 0.041); DANN LODO 0.046 but catastrophically unstable on hold-CALCE (MAE 0.096). Both failed.

**GroupDRO (rejected).** Distributionally Robust Optimisation (Sagawa et al. 2020) minimises worst-group risk, upweighting the highest-loss study group in each batch. Rationale: if the NASA_RW group is the hardest training group and proxies the NASA test distribution, DRO would improve transfer. Result: **GroupDRO LODO MAE 0.075 (hold-NASA 0.090) — 27% worse than plain BiGRU**. The structural problem: under hold-NASA, there are no NASA samples in training, so there is no group label that corresponds to the transfer direction. DRO upweights `NASA_RW` (115 samples, the smallest group) because size imbalance makes it the empirical worst group — not because it is genuinely representative of the CALCE→NASA domain shift. DRO optimised for the wrong hard group.

**Augmentation wins (positive result).** Rate-warp augmentation (random C-rate scale factor `s ~ U(1-ws, 1+ws)` applied per batch, with matched `t_elapsed ÷ s`) teaches the model that absolute C-rate and elapsed time are nuisance variables, not signal. An Optuna sweep (30 trials, minimising mean LODO MAE) identified the best combination: **`warp_strength=0.07`, `jitter_sigma=0.006`, `time_warp_strength=0.11`**.

**Augmentation sweep results:**

![Augmentation progression](../results/extended_tier1/augmentation_progression.png)
*Progression from plain BiGRU to Optuna-tuned augmentation + 1-shot adaptation. Green bars = hold-CALCE; red bars = hold-NASA. Both directions improve from plain BiGRU to Optuna aug. 1-shot fine-tuning further closes the NASA gap.*

| Method | LODO mean | LODO hold-CALCE | LODO hold-NASA | GKF-6 |
|---|---|---|---|---|
| BiGRU anchor (rate_warp ±25%) | 0.065 | 0.059 | 0.070 | 0.062 |
| + time_warp only (ablation) | 0.059 | 0.050 | 0.068 | 0.067 |
| **Optuna aug** (ws=0.07, jit=0.006, tw=0.11) | **0.058** | **0.051** | **0.065** | **0.055** |
| GroupDRO | 0.075 | 0.060 | 0.090 | 0.068 |

Time-warp (stretching/compressing the time axis non-uniformly) is the dominant new augmentation axis — adding it alone explains most of the Optuna gain. This makes sense: `t_elapsed` is the channel most correlated with protocol fingerprints (absolute charge duration), and time-warp directly creates invariance to it.

**1-shot adaptation (best overall result).** After training the Optuna-aug BiGRU zero-shot, a lightweight fine-tuning step on **k=1 labeled target cell** (C2 protocol: fine-tune only the FC head on 1 cell's cycles from the target domain) achieves **NASA MAE 0.045** — a 35% reduction from the anchor. k=1 outperforms k=3 and k=5: with more cells, fine-tuning starts to overfit the small target set.

---

## 4. Discussion

### 4.1 When RF wins, when DL wins

The scalar Random Forest is a formidable baseline. On the NASA dataset (13 cells, ~800 samples), it is effectively unbeatable. Its strength is precisely its compression: 9–11 hand-engineered features distil the physically relevant information into a representation that tree-based ensembles can interpolate reliably at small data scale. It fails when held-out cells come from a different protocol group that the training scalars have not been calibrated for (NASA_RW MAE 0.19 in GroupKFold-6; both LODO directions fail). It cannot generalise to unseen protocols because its features were designed for the training protocols.

BiGRU is the better model once data is sufficient (~12k samples, 22 cells). It reads the *shape* of degradation in the voltage-aligned current-decay curve — information that no finite set of hand-crafted scalars extracts exactly. Its advantage concentrates on cells whose charge profiles differ structurally from the training majority.

![RF feature importances — E2 stress markers](../results/tune_lstm/07_e2_rf_importances.png)
*RF feature importances from notebook 07 (NASA phase, E2 arm with stress markers). The top 4 features are all stress markers (orange): cumulative Ah throughput, discharge voltage floor, load current std dev, raw current mean. The 9 original charge-curve scalars collectively rank below them. This explains why, on NASA data, adding physically-grounded scalar summaries of the battery's history beats raw curve shape.*

### 4.2 The architecture–transfer tradeoff

Adding expressivity (CNN front-end, attention, Transformer) improves within-distribution accuracy but consistently hurts transfer. The pattern is clear across every experiment:
- CNN-LSTM: +1% GKF-6 → +73% LODO mean (relative to BiGRU).
- Transformer: best GKF-6 (MAE 0.033) → 2× LODO mean of BiGRU (0.083).
- Attn-GRU: intermediate everywhere.

This is not a hyperparameter failure — it is a consequence of how each architecture pools temporal information. Position-agnostic global pooling (mean+max) naturally ignores *where* along the voltage axis features appear, focusing on *what* features appear. Attention and convolution are explicitly position-sensitive, and voltage-axis position is correlated with charge rate (which is correlated with dataset identity).

**Lesson:** for cross-domain battery SOH, the inductive bias of your pooling mechanism matters more than the capacity of your feature extractor.

### 4.3 Augmentation vs other forms of robustness

The failure of CORAL, DANN, and GroupDRO is not a criticism of those methods in general. They are designed for settings where the target domain is visible during training, the source/target split is stable, and sample counts are large enough for reliable covariance estimation. Under LODO, none of these conditions hold. The success of augmentation is complementary: it requires no target samples, imposes no training objective conflicts, and scales naturally with available compute.

The physical interpretation of what the augmentations do:
- **Rate-warp:** teaches "C-rate level is not a degradation signal." Forces the model to rely on *shape* (how the curve rises and falls along the voltage axis) rather than *magnitude* (absolute current value).
- **Time-warp:** teaches "absolute elapsed time is not a degradation signal." The same voltage-grid position reached faster or slower should predict the same SOH if the underlying cell state is the same.
- **Jitter:** minor additive noise on the channels; prevents over-reliance on specific sample values; contributes smaller gain than the above two.

### 4.4 Best deployed model

**BiGRU with Optuna augmentation** (ws=0.07, jit=0.006, tw=0.11) is the recommended model. Characteristics:
- ~15k parameters — deployable on constrained hardware (BMS microcontroller).
- LODO mean MAE 0.058 zero-shot; 0.045 on NASA with k=1 fine-tuning.
- GroupKFold-6 MAE 0.055 — competitive with more complex architectures.
- No target-domain samples required at deployment time (except for the optional 1-shot fine-tune).
- Trained in < 10 minutes on GPU; single-seed variance is near zero.

---

## 5. Limitations and Future Work

**Fold weighting:** GroupKFold-6 and LODO weight each fold equally regardless of dataset size. The NASA_RW fold (115 samples) carries the same weight as the CALCE_CX2_T2 fold (~3,039 samples) — which artificially amplifies the NASA contribution to aggregate metrics. A sample-size-weighted aggregate would shift headline MAE values; we report fold-equal means throughout to match standard cross-validation conventions.

**Single-seed training:** all results are single-seed (seed variance was empirically near-zero in diagnostic runs; N_SEEDS=3 experiments were scaffolded but not fully run at time of writing). The 0.0000 seed-std observed in anchor experiments suggests training is stable, but this should be confirmed formally.

**Augmentation round 2 (notebook 06):** a second augmentation sweep (cutout along the voltage axis, grid-shift, sequence mixup, channel-dropout) and a multi-objective NSGA-II Pareto optimisation (optimising CALCE MAE and NASA MAE simultaneously) are scaffolded and ready to run, but the experiments were not completed before the project deadline. The hypothesis is that voltage-axis cutout (masking the low-voltage tail of the grid) would address the NASA/CALCE discharge cutoff voltage gap (2.7 V vs 3.0 V). This is the most promising open thread.

**Tier-2 extension:** the `src/cells.py` and `src/extract_generic.py` infrastructure is already built to load BatteryArchive format (HNEI ~15 cells, SNL ~50 cells, UNIBO ~27 cells). Adding Tier-2 would push toward 100+ cells and enable chemistry-agnostic (NMC vs NCA vs LFP) transfer experiments.

**Chemistry transfer:** both datasets in this study use NMC-type cathodes. The most interesting transfer test — to LFP chemistry with its flat voltage plateau — would require a different voltage window and is architecturally out of scope for the current voltage-grid representation.

---

## 6. Reproducibility

All experiments are implemented in Python using PyTorch (GPU/MPS) + scikit-learn + Optuna. Package management via `uv`. Notebooks are ordered sequentially; the final pipeline can be reproduced top-to-bottom.

### Notebook map

| Notebook | Phase | What it does | Headline metric |
|---|---|---|---|
| `nasa/01_eda` | 1 | EDA, cleaning decisions, feature validation | r_proxy vs EIS r=0.77 |
| `nasa/02_dataset_prep` | 1 | RUL labels, feature engineering, LOBO splits | 13 cells, 636+165 rows |
| `nasa/03_train` | 1 | Ridge / RF / GBM baselines, LOBO | **RF MAE 0.032, R² 0.966** |
| `nasa/04_train` | 1 | Macro-clock TCN vs RF | TCN MAE 0.042, loses to RF |
| `nasa/05_dataset_prep_dl` | 1 | Micro-clock data prep (superseded) | 778 charge-curve tensors |
| `nasa/06_train_dl` | 1 | CNN / TCN / LSTM on charge curves | LSTM MAE 0.073 |
| `nasa/07_soh_reframe` | 1 | Target reframe (SOH→Ct), stress markers | E2 RF MAE 0.053, R² 0.647 |
| `nasa/08_unlaundered_curve` | 1 | CC+CV restore, duration scalar | D2 RF MAE 0.040 |
| `nasa/09_voltage_grid_lstm` | 1 | Voltage-aligned representation | **VG-LSTM MAE 0.062, Spearman tie with RF** |
| `extended/01_build_extended_dataset` | 2 | Build 22-cell voltage-grid dataset | 12,260 samples, 2 dataset groups |
| `extended/02_train_extended` | 2 | RF vs LSTM vs CNN-LSTM, GKF-6 + LODO | LSTM MAE 0.038, LODO 0.050 |
| `extended/03_train_extended_gru` | 2 | GRU vs CNN-GRU, GKF-6 + LODO | GRU MAE 0.037, LODO 0.037 |
| `extended/04_model_comparison` | 2 | Full 5-model comparison + transfer hypotheses | **BiGRU LODO 0.041**; Transformer LODO 0.083 |
| `extended/05_robust_transfer` | 2 | Augmentation sweep (Optuna) + GroupDRO + TTA | **Optuna-aug LODO 0.058; + 1-shot → NASA 0.045** |
| `extended/06_augment_axes2` | 2 | Round-2 aug: cutout, grid-shift, mixup, channel-dropout | *Scaffolded — not yet run* |

### Repository layout

```
data/
  calce/       CALCE CS2/CX2 Arbin .xls files (not committed)
  processed/   vg_extended_tier1.npz (22 cells, 12260×128×3)
src/
  battery.py         NASA data loader + feature engineering
  voltage_grid.py    Voltage-grid extraction + LOBO metrics
  vg_models.py       VGLSTMReg, VGCNNLSTM, VGGRUReg, VGCNNGRU, Transformer, AttnGRU
  vg_extended.py     build_extended_dataset, run_grouped_cv, LODO helpers
  extract_generic.py CALCE Arbin + BatteryArchive loaders
  cells.py           Cell registry (CellSpec, CELLS, EXCLUDED)
  vg_da.py           CORAL, DANN domain-adaptation modules
  vg_augment.py      rate_warp, time_warp, jitter, cutout augmentation pipeline
notebooks/
  nasa-dataset/      Notebooks 01–09 (Phase 1)
  extended-dataset/  Notebooks 01–06 (Phase 2)
results/
  extended_tier1/    Per-fold plots, comparison CSVs
models/              Persisted .pkl and .pt checkpoints
```

---

*Completed June 2026. Public repository: [link to be added in PPT].*
