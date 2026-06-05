# Extended Dataset Pipeline — Design & Data Specification

> **Replaces** the stale `../nasa-dataset/05_dataset_prep_dl_summary.md`.
> That summary describes an earlier design. The live code (`src/voltage_grid.py`)
> differs significantly; this document reflects the current implementation and the
> multi-dataset extension plan.

---

## 1. Current Pipeline — What `src/voltage_grid.py` Does Today

### The voltage-grid representation

For each retained discharge cycle, the pipeline extracts the **CC charging phase**
immediately preceding it and resamples it to a fixed-length tensor on a
**descending voltage axis**.

| Parameter | Value | Reason |
|---|---|---|
| Voltage window | 3.8 → 4.2 V (descending) | Full CC window ending at the CV knee (4.2 V). Descending places valid data at the sequence start; truncation at the tail is the standard right-padding convention — compatible with masked pooling. |
| Grid resolution | N = 128 pts | Balances curve-shape fidelity vs model size. |
| Channels | `[|I|, T, t_elapsed]` | Voltage is the *axis*, not a channel. `|I|` carries charge-rate information; `T` adds thermal context; `t_elapsed` encodes time-to-charge-completion as a degradation proxy. |
| Target | `Ct / C_nominal` | Dataset-agnostic denominator — does not require a per-battery measured baseline (unlike `capacity_retention`), enabling multi-dataset merging. For NASA: `C_nominal = 2.0 Ah` (controlled) / `2.2 Ah` (RW). |
| Mask | Leading boolean array, shape `(n_grid,)` | `True` = valid grid position. The tail (low-V end) is zero-filled for cycles that didn't charge from below V_LO. Compatible with masked mean+max pooling in `vg_models.VGCNNLSTM`. |

### Why the 05 summary is stale

`notebooks/nasa-dataset/05_dataset_prep_dl_summary.md` was written during an earlier
iteration:

| Field | 05 summary (stale) | Live code (correct) |
|---|---|---|
| Voltage window upper bound | 4.1 V | **4.2 V** (`V_HI = 4.2`) |
| Channels | `[V, I, T]` | **`[|I|, T, t_elapsed]`** — voltage is the axis |
| Target | `capacity_retention` (per-battery baseline) | **`Ct / C_nominal`** (rated capacity) |
| NPZ key for target | `soh` | **`y`** |

The live code is the authoritative source. Do not use the 05 summary for reference.

---

## 2. Why We're Extending

13 NASA cells (~700 samples) is too small to clearly separate the CNN-LSTM from
tree baselines. Expanding to 25+ cells (Tier 1) and 100+ cells (Tier 2) gives:

1. Enough data to see whether the DL model's within-cycle curve-feature advantage
   shows up in held-out metrics vs a random forest on the same scalars.
2. A **chemistry/dataset transfer test** (Leave-One-Dataset-Out) — a stronger claim
   for the presentation than single-dataset LOBO.

### RW9 correction

RW9 belongs to the random-SOC group RW9–RW12. The papers drop this group because
these batteries start at arbitrary SOH. We had kept only RW9 by mistake. It is now
removed, leaving **8 RW cells**: RW1, RW13–RW17, RW19, RW20. This matches the
"~8 after cleaning" in the dataset table.

---

## 3. Data Format Specifications per Source

### NASA PCoE Li-Ion (existing — `.mat` format)

| Field | Value |
|---|---|
| File format | MATLAB `.mat`, loaded via `scipy.io.loadmat` (→ `battery.load_controlled` / `load_randomized`) |
| Charge fields | `Voltage_measured (V)`, `Current_measured (A)`, `Temperature_measured (°C)`, `Time (s)` |
| Cycle ID | Sequential `type == 'charge'` / `type == 'discharge'` entries |
| Nominal capacity | 2.0 Ah (controlled B0005–B0018); 2.2 Ah (randomized RW series) |
| Per-sample temperature | **Yes** |
| Group (LODO) | `"nasa"` |
| Loaders | `battery.load_controlled`, `battery.load_randomized` (unchanged) |
| Extractors | `voltage_grid.extract_controlled_vg`, `voltage_grid.extract_randomized_vg` (unchanged) |

**Retained cells:** B0005, B0006, B0007, B0018 (controlled);
RW1, RW13–RW17, RW19, RW20 (randomized, 8 cells).

---

### CALCE CS2 / CX2 (new — Arbin `.xls` format)

| Field | Value |
|---|---|
| File format | Multiple Arbin `.xls` files per cell (one per test session), concatenated |
| Cycle column | `Cycle_Index` |
| Current column | `Current(A)` — positive = charge, negative = discharge |
| Voltage column | `Voltage(V)` |
| Time column | `Test_Time(s)` |
| Discharge capacity | `Discharge_Capacity(Ah)` — cumulative within each file, delta per cycle taken as max−min over the largest contiguous negative-current run |
| Temperature | **Not logged** — impute constant `T = 25 °C` (CALCE stated ambient) |
| Nominal capacity | 1.1 Ah |
| Charge protocol | CC-CV, 0.5C charge to 4.2 V |
| Per-sample temperature | **No** |
| Group (LODO) | `"calce"` |
| Loader | `extract_generic.read_calce_cell_dir` |

**Expected directory structure after download:**

```
data/calce/
├── CS2/
│   ├── CS2-33/        ← one folder per cell, N .xls files inside
│   ├── CS2-34/
│   ├── CS2-35/
│   ├── CS2-36/
│   ├── CS2-37/
│   └── CS2-38/
└── CX2/
    └── {cell-id}/     ← verify exact IDs against downloaded files
```

**CS2 excluded:** CS2_8, CS2_21 — tested on CADEX cycler (different column format,
incompatible with the Arbin reader).

> **Parsing fixes applied (2026-06-05)** — two bugs were found and fixed in
> `src/extract_generic.py`:
>
> 1. **Chronological file ordering.** CALCE filenames use an unpadded `M_D_YY` scheme
>    (e.g. `CS2_35_10_15_10.xlsx`) which does not sort correctly as strings — August
>    files sorted after January, placing early high-capacity cycles at the *end* of the
>    sequence and producing jagged, non-monotone SOH trajectories.  Fix: files are now
>    ordered by the first `Date_Time` value recorded inside each file
>    (`_parse_first_datetime` in `extract_generic.py`), with a filename-date fallback.
>
> 2. **Implausible per-cycle capacity band.** `Discharge_Capacity(Ah)` is cumulative
>    within each Arbin file. Some `Cycle_Index` entries aggregate many sub-cycles into
>    one group (e.g. CX2_16 cycle 43 = 12.5 Ah → SOH ≈ 11×). A plausibility band of
>    `[0.2 × C_nom, 1.5 × C_nom]` = `[0.22, 1.65]` Ah is applied after computing the
>    per-cycle delta; cycles outside the band are set to NaN and excluded by
>    `attach_targets_extended`. Measured impact across all 11 CALCE cells: **1.3%**
>    of 14,391 cycles dropped (181 partial/characterisation cycles below the lower
>    bound + 1 malformed cycle above the upper bound).

**CX2 excluded** (verify exact IDs at download):
- **CX2_16** — hard protocol change at cycle 1152: first 1114 cycles at −0.55 A (0.5C),
  remaining 894 cycles at −0.675 A (0.61C). The two phases are physically
  incomparable on a single SOH scale; the step at cycle 1152 is a test-protocol
  artifact, not degradation.
- Cells tested on CADEX cycler
- Cell(s) cycled at varying temperatures (25–55 °C) — multi-T without T as input
  feature creates unlearnable variance
- Cell(s) with pulsed-discharge protocol (Type 4: alternating 0.5C/1C with rest)
- Cell(s) with partial-cycling protocol (Type 6)

> ⚠️ Add confirmed CX2 cell IDs to `src/cells.py` after downloading.

---

### BatteryArchive (Tier 2 — `_timeseries.csv` format)

| Field | Value |
|---|---|
| File format | Standard BatteryArchive `_timeseries.csv` per cell |
| Cycle column | `Cycle_Index` |
| Current column | `Current(A)` |
| Voltage column | `Voltage(V)` |
| Time column | `Test_Time(s)` |
| Temperature column | `Cell_Temperature(C)` |
| Discharge capacity | `Discharge_Capacity(Ah)` |
| Per-sample temperature | **Yes** |
| Loader | `extract_generic.read_batteryarchive_csv` |

**HNEI** — NMC-LCO 18650, ~2.8 Ah, 4.2 V, 15 cells, ~1000 cycles, C/2 charge /
1.5C discharge, 25 °C. Group: `"hnei"`.

**SNL** — Commercial 18650 NCA + NMC + LFP. **Drop LFP cells** (3.6 V plateau,
different window). ~50 NCA/NMC cells usable. Group: `"snl"`.

**UNIBO Powertools** — NMC 18650, mixed manufacturers/capacities, 27 cells, CC-CV
1.8A to 4.2V. Group: `"unibo"`.

> ⚠️ Phase 2: add cell entries to `src/cells.py` after downloading and confirming
> `Cell_Temperature` column is populated.

---

## 4. Representation Decisions for Multi-Dataset Merging

All decisions confirmed during the brainstorm phase:

| Decision | Choice | Rationale |
|---|---|---|
| Current channel | **C-rate** `\|I\| / C_nom` | Raw Amps confound cell size with charge rate (CALCE 0.55 A ≈ NASA 1.5 A, both ~0.5C). C-rate makes curves comparable across 1.1–3.2 Ah cells. |
| Temperature imputation | **Stated ambient** (CALCE = 25 °C; unknown = 24 °C) | Not the training mean — physically honest. Scales to ≈ neutral through the same `StandardScaler` as real T. |
| Target | **`Ct / C_nom`** per cell | `C_nom` from the cell spec sheet, not a measured baseline. NASA controlled 2.0, NASA RW 2.2, CALCE 1.1, HNEI 2.8. |
| RW9 | **Fully dropped** | Random-SOC group (RW9–RW12); papers drop this group. |
| CV scheme | **GroupKFold-8** (headline) + **LODO** (transfer) | GroupKFold holds out whole cells in ~8 folds (fast, LOBO semantics). LODO tests cross-chemistry generalisation — a stronger claim. |

---

## 5. Training Process

### Normalization (applied per cell during extraction, before caching to .npz)

1. **C-rate normalisation**: `tensor[:, 0] /= C_nom` — applied in the extractor after
   calling `resample_voltage_grid`. The existing function is not modified.
2. **Temperature imputation**: for cells without per-sample T (`has_temp=False`),
   all rows of the T channel are set to `ambient_c` before calling
   `resample_voltage_grid`.
3. **Per-channel `StandardScaler`**: fit on masked-valid training rows only
   (`voltage_grid.global_scale`). After scaling, zero-fill is re-applied to
   uncovered (mask=False) positions via `voltage_grid._reapply_fill`.
4. **Target**: `y = capacity_ahr / C_nom`. Expected range ≈ [0.75, 1.05]
   (≤ 0.75 = near/past EOL, > 1.0 = fresh cell slightly above rated).

### Model — CNN-LSTM (`src/vg_models.VGCNNLSTM`)

```
Input (B, 128, 3)   channels: [|I|/C_nom,  T (°C),  t_elapsed (s)]
  └─ Conv1d(3→8, k=5, pad=2) + GroupNorm(4) + ReLU + Dropout(0.35)
  └─ BiLSTM(hidden=40, bidirectional)
  └─ masked mean+max pool  →  (B, 160)
  └─ Dropout(0.35)
  └─ Linear(160 → 1)
  └─ clip [0.0, 1.1]
```

- **GroupNorm** instead of BatchNorm: stable when held-out-battery folds have
  different statistics than the batch (which is the common case in LOBO/GroupKFold).
- **No `pack_padded_sequence`**: padded positions hold zero (= scaled channel mean)
  and masked pooling ignores them. Packing forces a GPU→CPU sync per batch —
  measurably worse performance for small models on MPS.

### Cross-validation (`src/vg_extended.run_grouped_cv`)

**Headline — GroupKFold-8:**

```python
from sklearn.model_selection import GroupKFold
splitter = GroupKFold(n_splits=8)
folds = run_grouped_cv(model_factory, X, mask, y, groups, cidx, device, splitter)
```

- `groups` = `battery_id` per sample → whole cells always stay together.
- Inner-val: last-alphabetical battery in the train split (deterministic,
  dataset-agnostic — no hard-coded NASA cell names).
- Scale fit on masked-valid **train rows only** (no leakage from val/test).

**Transfer — Leave-One-Dataset-Out (LODO):**

```python
from sklearn.model_selection import LeaveOneGroupOut
splitter = LeaveOneGroupOut()
folds = run_grouped_cv(..., splitter, cv_groups=ds_groups)
```

- `ds_groups` = dataset-level group label (`"nasa"`, `"calce"`, `"hnei"`, …).
- Each fold holds out one entire dataset/chemistry — tests cross-dataset transfer.
- Inner-val: same last-alphabetical-battery-in-train logic.

### Metrics

MAE, RMSE, R², skill score (1 − MAE / naive_MAE), Spearman ρ.
Reported as mean ± std across GroupKFold-8 folds; per-dataset for LODO.
Functions: `voltage_grid.lobo_metrics`, `voltage_grid.aggregate` (unchanged).

---

## 6. New Source Files

| File | Role |
|---|---|
| `src/cells.py` | Cell registry (`CellSpec` dataclass + `CELLS` list + `EXCLUDED` dict) |
| `src/extract_generic.py` | Adapters for CALCE Arbin `.xls` and BatteryArchive `.csv`; reuses `resample_voltage_grid` without modification |
| `src/vg_extended.py` | `build_extended_dataset`, `attach_targets_extended`, `save/load_npz_ext`, `run_grouped_cv` — all new; imports from `voltage_grid` but does not modify it |

Existing files (`src/voltage_grid.py`, `src/vg_models.py`, `src/sequence.py`,
`src/battery.py`) are **not modified**.
