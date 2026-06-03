# Training Summary — `03_train.ipynb`

## Modeling Framework

Two phases capturing different deployment realities:

| Phase | Data | Resistance features | LOBO folds |
|---|---|---|---|
| **Oracle** | Controlled only (4 batteries) | EIS `Re`/`Rct` + `r_proxy` | 4 |
| **Practical** | Controlled + Randomized (13 batteries) | `r_proxy` only | 13 |

**Validation:** Leave-One-Battery-Out (LOBO) cross-validation — each fold holds out one complete battery.

**Targets:** `rul_frac` (lifespan, 0–1) and `energy_retention` (efficiency, 0–1).

**Hyperparameter tuning:** Optuna (100 trials, minimise LOBO MAE) on oracle phase; manual baselines on practical phase.

## Results

### Oracle Phase (4 batteries)

**Lifespan (`rul_frac`):**

| Model | MAE ± std | RMSE ± std | R² ± std | MAE (cycles) | Monotonicity |
|---|---|---|---|---|---|
| Ridge | 0.1057 ± 0.034 | 0.1302 ± 0.042 | 0.816 ± 0.110 | 8.7 ± 1.0 | 0.730 |
| Random Forest | 0.0614 ± 0.013 | 0.0933 ± 0.027 | 0.908 ± 0.049 | 5.2 ± 0.7 | 0.850 |
| **GBM** | **0.0565 ± 0.019** | **0.0855 ± 0.028** | **0.925 ± 0.048** | **5.1 ± 2.0** | 0.597 |

**Efficiency (`energy_retention`):**

| Model | MAE ± std | RMSE ± std | R² ± std | Monotonicity |
|---|---|---|---|---|
| **Ridge** | **0.0196 ± 0.006** | **0.0253 ± 0.008** | **0.932 ± 0.053** | 0.737 |
| Random Forest | 0.0324 ± 0.020 | 0.0378 ± 0.022 | 0.859 ± 0.108 | 0.700 |
| GBM | 0.0280 ± 0.023 | 0.0333 ± 0.024 | 0.884 ± 0.111 | 0.617 |

### Practical Phase (13 batteries)

**Lifespan (`rul_frac`):**

| Model | MAE ± std | RMSE ± std | R² ± std | MAE (cycles) | Monotonicity |
|---|---|---|---|---|---|
| Ridge | 0.1358 ± 0.078 | 0.1710 ± 0.085 | 0.560 ± 0.779 | 3.1 ± 3.3 | 0.739 |
| **Random Forest** | **0.0317 ± 0.019** | **0.0537 ± 0.028** | **0.966 ± 0.033** | **1.5 ± 2.3** | **0.926** |
| GBM | 0.0349 ± 0.023 | 0.0556 ± 0.033 | 0.961 ± 0.038 | 1.8 ± 2.6 | 0.737 |

**Efficiency (`energy_retention`):**

| Model | MAE ± std | RMSE ± std | R² ± std | Monotonicity |
|---|---|---|---|---|
| Ridge | 0.0574 ± 0.030 | 0.0686 ± 0.029 | 0.757 ± 0.209 | 0.846 |
| **Random Forest** | **0.0323 ± 0.035** | **0.0373 ± 0.038** | **0.848 ± 0.243** | **0.855** |
| GBM | 0.0373 ± 0.037 | 0.0442 ± 0.042 | 0.811 ± 0.293 | 0.813 |

## Key Insights

- **GBM oracle lifespan:** Best MAE (5.1 cycles, ~3–5% of battery life) but poor monotonicity (0.597) — predicts false improvements within a cycle sequence.
- **Ridge oracle efficiency:** Wins because energy retention is a smooth, approximately linear decline; linear model exploits this structure.
- **RF practical outperforms RF oracle:** More training batteries (13 vs 4) stabilise bagging; `rul_frac` normalisation enables cross-battery generalisation.
- **RF monotonicity (0.926):** Bagging smooths cycle-to-cycle oscillations better than boosting.

## Persisted Models (`models/`)

| File | Algorithm | Phase | Task |
|---|---|---|---|
| `oracle_lifespan_gbm.pkl` | GBM (n_est=376) | Oracle | Lifespan |
| `oracle_efficiency_ridge.pkl` | Ridge (α=98.9) | Oracle | Efficiency |
| `practical_lifespan_rf.pkl` | RF (n_est=300) | Practical | Lifespan |
| `practical_efficiency_rf.pkl` | RF (n_est=300) | Practical | Efficiency |

Each `.pkl` includes `model`, `feature_cols`, `phase`, `task`, `algorithm` metadata.

## Top Features (RF, by task)

**Lifespan (both phases):** `capacity_retention`, `r_proxy_5_roll_mean`, `capacity_ahr`, `r_proxy_10_roll_std`, `temp_mean_5_roll_std`

**Efficiency (both phases):** `r_proxy` (direct onset DCIR), `r_proxy_5_roll_std`, `voltage_mean`, `temp_mean`, `duration_s`
