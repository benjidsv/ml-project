# DL Training Summary — `04_train.ipynb`

## What we did

First deep learning notebook of the project. Trained a **TCN (Temporal Convolutional Network)**
on the *macro clock* — one timestep = one full discharge cycle, sequence = the entire
degradation trajectory — and ran an apples-to-apples LOBO rematch against the Practical RF
baseline from notebook 03.

**Setup:**

| | |
|---|---|
| Phase | Practical (13 batteries, `r_proxy` only — no EIS) |
| Targets | `rul_frac` (lifespan, lead) · `energy_retention` (efficiency, secondary) |
| Architecture | TCN — 2 causal dilated conv blocks, 32 channels, ~4 k params |
| Lookback L | 5 cycles (selected by tier-1 sweep over {5, 10, 15, 20}) |
| Validation | Leave-One-Battery-Out, 13 folds — same protocol as notebook 03 |
| Features | 8 base per-cycle scalars (no pre-rolled stats — the sequence re-derives temporal structure) |

**Two-tier training:**
1. **Tier-1 (fixed split):** swept L × {LSTM, GRU, TCN} on the notebook-03 train/test split to pick the best architecture and epoch budget. TCN converged fastest (16 epochs, val MAE 0.031) and won clearly over LSTM (124 epochs, 0.048) and GRU (224 epochs, 0.043).
2. **Tier-2 (LOBO):** ran full 13-fold LOBO with TCN at L=5 for both targets.

## Results

### Lifespan (`rul_frac`) — TCN vs RF

| Model | MAE ± std | RMSE | R² ± std | MAE (cycles) | Monotonicity | Skill |
|---|---|---|---|---|---|---|
| **RF** (baseline) | **0.0317 ± 0.019** | **0.0537** | **0.966 ± 0.033** | **1.5** | **0.926** | **0.892** |
| TCN (DL macro) | 0.0417 ± 0.021 | 0.0702 | 0.935 ± 0.078 | 1.6 | 0.847 | 0.858 |

### Efficiency (`energy_retention`) — TCN vs RF

| Model | MAE ± std | RMSE | R² ± std | Monotonicity | Skill |
|---|---|---|---|---|---|
| **RF** (baseline) | **0.0323 ± 0.035** | **0.0373** | **0.848 ± 0.243** | **0.855** | **0.758** |
| TCN (DL macro) | 0.0570 ± 0.032 | 0.0718 | 0.592 ± 0.570 | 0.697 | 0.560 |

## Key Insights

- **TCN loses to RF on lifespan.** MAE is 32% higher (0.042 vs 0.032) and monotonicity lower (0.847 vs 0.926). The cycle-level MAE is nearly identical (1.6 vs 1.5 cycles) — the degradation *level* is captured, but the TCN trajectory is noisier fold-to-fold. We observe that the beginning of the curve isn't accurate, which makes sense - it's the sequence's cold start.
- **TCN clearly loses on efficiency.** R² 0.592 ± 0.570: the fold variance equals the mean, meaning several folds reach near-zero or negative R². The efficiency signal is too noisy for a macro-clock model with only 13 trajectories.
- **Fast convergence = data starvation signal.** TCN stopped at epoch 16 vs LSTM at 124. The model ran out of signal quickly, not training time — consistent with 13 trajectories being fundamentally insufficient for a sequence model.
- **Root cause is not the architecture, it's the data count.** 13 trajectories limits what any sequence model can learn at the macro level. RF's bagging provides better implicit regularisation at this scale.

## Persisted Model (`models/`)

| File | Architecture | Phase | Task | L |
|---|---|---|---|---|
| `practical_lifespan_tcn_L5.pt` | TCN | Practical | Lifespan | 5 |

Each `.pt` includes `state_dict`, `channels`, `lookback`, `scaler_mean`, `scaler_scale`, and metadata.

## What's next

The macro-clock loss is the intended bridge to the within-cycle approach:

- The scalar baselines (and this TCN) reduce each charge curve to a handful of aggregates.
  They never see the *shape* of the curve — how voltage rises through the CC phase, how the
  plateau shifts as the cell ages.
- Moving to the **micro clock** (notebooks 05–06) changes both the data count (13 → ~800
  charge-curve segments) and the information available (raw V/I/T time-series per cycle).

**Notebook 05** — extract and preprocess the within-cycle charge-curve tensors from the raw
`.mat` files: voltage-range slicing (3.8–4.1 V), resample to fixed length, 3-channel (V, I, T)
tensors, LOBO-compatible battery-level splits.

**Notebook 06** — train 1D-CNN / TCN / LSTM on the charge-curve tensors to estimate per-cycle
SOH directly. With ~800 samples and a signal the baselines never accessed, this is where DL
is expected to earn its place.
