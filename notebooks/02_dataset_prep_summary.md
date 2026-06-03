# Dataset Prep Summary — `02_dataset_prep.ipynb`

## Batteries Retained (13 total)

**Excluded:**
- Hot (B0029–B0032): never reach 80% EOL threshold over 40 cycles
- Cold (B0042–B0044): internal inconsistency (B0042 → 0.889 Ahr vs B0043/44 → 1.49 Ahr)
- RW2, RW18: sensor errors (temperature readings all −4093°C)

**Retained:**
- Controlled: B0005, B0006, B0007, B0018 (636 rows)
- Randomized: RW1, RW9, RW13–RW17, RW19–RW20 (165 rows)

## RUL Labels (Physically Grounded)

EOL threshold: 80% capacity retention (industry standard; every retained battery crosses it).

| Label | Definition |
|---|---|
| `rul_cycles` | Remaining cycles to first sub-80% crossing (clipped ≥ 0 after EOL) |
| `rul_frac` | `rul_cycles / eol_index` ∈ [0, 1] — normalised across batteries with different lifespans |

EOL indices: B0005→99, B0006→60, B0007→123, B0018→76; RW batteries→4–11 cycles (rapid degradation).

## Feature Engineering (Causal, No Look-Ahead)

| Family | Features | Windows |
|---|---|---|
| Rolling statistics | Mean/std of `voltage_mean`, `temp_mean`, `r_proxy` | 5, 10, 15, 20 cycles |
| Capacity fade rate | `capacity_ahr.diff()` rolling mean | 5, 10, 15, 20 |
| Resistance rate | `r_proxy.diff()` rolling mean | 5, 10, 15, 20 |
| Cycle position | `cycle_norm = cycle_index / max_cycle` | — |
| Capacity delta | `capacity_ahr − battery_baseline` | — |

**Efficiency target:** `energy_retention = cycle_Wh / baseline_Wh` (captures voltage sag, not just capacity loss).

**Leakage guards:** `capacity_ahr`, `capacity_retention`, `cap_fade_rate`, `cap_delta_from_baseline` excluded from efficiency models. `voltage_min` excluded from RW models (constant 3.2V floor — no signal).

## Train/Test Split (Whole-Battery Hold-Out)

Splitting by time within a battery leaks early-life context into test; always hold out entire batteries.

| Dataset | Train | Test |
|---|---|---|
| Controlled | B0005, B0006, B0018 | B0007 |
| Randomized | RW1, RW9, RW13, RW14, RW16, RW17 | RW15, RW19, RW20 |

## Output CSVs

| File | Rows | Features | Targets | Resistance |
|---|---|---|---|---|
| `data/controlled.csv` | 636 | 22 cols | `rul_cycles`, `rul_frac`, `energy_retention` | EIS `Re`, `Rct` + `r_proxy` |
| `data/randomized.csv` | 165 | 18 cols | same | `r_proxy` only |
