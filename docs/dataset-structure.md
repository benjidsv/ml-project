# Dataset Structure

All data is sourced from the **NASA Prognostics Center of Excellence (PCoE) Li-ion Battery Aging datasets**.
Controlled files are MATLAB `.mat` format containing a `cycle` struct array. Randomized files are available in both `.mat` and `.Rda` (R) formats containing a `step` struct array.

---

## Directory Overview

```
data/
├── controlled/                    # Fixed charge/discharge profiles (28 batteries, B0005–B0056)
│   ├── 1. BatteryAgingARC-FY08Q4/          # Primary baseline (B5–B7, B18) — room temp
│   ├── 2. BatteryAgingARC_25_26_27_28_P1/  # Square-wave discharge, room temp (B25–B28) [partial]
│   ├── 3. BatteryAgingARC_25-44/           # Multi-condition: room + hot + cold (B25–B44)
│   ├── 4. BatteryAgingARC_45_46_47_48/     # Cold temp, 1A discharge (B45–B48)
│   ├── 5. BatteryAgingARC_49_50_51_52/     # Cold temp, 2A discharge (B49–B52) [incomplete]
│   └── 6. BatteryAgingARC_53_54_55_56/     # Cold temp, 2A discharge (B53–B56)
│
└── randomized/                    # Random-walk current profiles (28 batteries, RW1–RW28)
    ├── Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/        # RW3–RW6
    ├── Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/  # RW1–RW2, RW7–RW8
    ├── Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/           # RW9–RW12
    ├── RW_Skewed_High_Room_Temp_DataSet_2Post/                                # RW17–RW20
    ├── RW_Skewed_High_40C_DataSet_2Post/                                      # RW25–RW28
    ├── RW_Skewed_Low_Room_Temp_DataSet_2Post/                                 # RW13–RW16
    └── RW_Skewed_Low_40C_DataSet_2Post/                                       # RW21–RW24
```

---

## Controlled Datasets

### Group 1 — `BatteryAgingARC-FY08Q4` (B0005–B0007, B0018)

| Battery | Discharge Cutoff | File Size |
|---|---|---|
| B0005 | 2.7 V | ~15 MB |
| B0006 | 2.5 V | ~15 MB |
| B0007 | 2.2 V | ~15 MB |
| B0018 | 2.5 V | ~8.1 MB |

- Temperature: room temperature
- Charge: CC at 1.5 A → CV until current drops to 20 mA (cutoff 4.2 V)
- Discharge: CC at **2 A** until voltage hits cutoff (varies per battery)
- Impedance: EIS sweep 0.1 Hz → 5 kHz
- EOL criterion: **30% capacity fade** (2.0 Ahr → 1.4 Ahr)

**Role in project:** Primary baseline. These are the batteries cited in CLAUDE.md as the canonical controlled-environment set for feature engineering and model comparison.

---

### Group 2 — `BatteryAgingARC_25_26_27_28_P1` (B0025–B0028)

- Temperature: 24 °C
- Discharge: **0.05 Hz square wave, 4 A amplitude, 50% duty cycle** (pulsed)
- Cutoff: 2.0 / 2.2 / 2.5 / 2.7 V respectively

> Partial/earlier copy of the same batteries in Group 3. Prefer Group 3 for completeness.

---

### Group 3 — `BatteryAgingARC_25-44` (B0025–B0044, excl. B0035 & B0037)

Five sub-groups with different temperature and discharge conditions:

| Sub-group | Temp | Discharge | EOL |
|---|---|---|---|
| B0025–B0028 | 24 °C | 4 A square wave | — |
| B0029–B0032 | **43 °C** | 4 A constant | 30% fade |
| B0033–B0036 | 24 °C | 4 A (B33/B34), 2 A (B36) | **20% fade** |
| B0038–B0040 | 24 °C & **44 °C** | 1 A, 2 A, 4 A (mixed) | 20% fade |
| B0041–B0044 | **4 °C** | 4 A & 1 A (mixed) | 30% fade |

B0033–B0036 files are notably larger (~11 MB) — more cycles were recorded before EOL.
B0041–B0044 README warns of anomalously low-capacity discharge runs with unanalyzed cause.

---

### Group 4 — `BatteryAgingARC_45_46_47_48` (B0045–B0048)

- Temperature: **4 °C** — Load: **1 A constant**
- Cutoff: 2.0 / 2.2 / 2.5 / 2.7 V — EOL: 30% fade
- Same anomalous low-capacity warning as Group 3 cold-temp sub-group

---

### Group 5 — `BatteryAgingARC_49_50_51_52` (B0049–B0052)

- Temperature: **4 °C** — Load: **2 A constant**
- Cutoff: 2.0 / 2.2 / 2.5 / 2.7 V
- **EOL: experiment ended when control software crashed — data is truncated**
- Anomalous low-capacity/low-voltage discharge runs present

---

### Group 6 — `BatteryAgingARC_53_54_55_56` (B0053–B0056)

- Temperature: **4 °C** — Load: **2 A constant**
- Cutoff: 2.0 / 2.2 / 2.5 / 2.7 V — EOL: 30% fade
- Same low-capacity anomaly warning

---

### Controlled `.mat` Data Structure

```
cycle[]  (one entry per operation)
├── type                   "charge" | "discharge" | "impedance"
├── ambient_temperature    °C
├── time                   MATLAB date vector (start of cycle)
└── data
    ├── [charge & discharge]
    │   ├── Voltage_measured      V
    │   ├── Current_measured      A
    │   ├── Temperature_measured  °C
    │   ├── Current_charge        A (at charger/load)
    │   ├── Voltage_charge        V (at charger/load)
    │   ├── Time                  s (time vector for cycle)
    │   └── Capacity              Ahr  ← discharge only; populated ONLY for 2.7 V cutoff cycles
    └── [impedance]
        ├── Sense_current         A
        ├── Battery_current       A
        ├── Current_ratio         (dimensionless)
        ├── Battery_impedance     Ω (raw)
        ├── Rectified_impedance   Ω (calibrated + smoothed)
        ├── Re                    Ω (electrolyte resistance)
        └── Rct                   Ω (charge transfer resistance)
```

> `Capacity` is only present on discharge cycles that hit **2.7 V**. For all other cutoffs it must be computed as `∫ current dt / 3600`.

---

## Randomized Datasets

All randomized datasets share the same `.mat` data structure and are also available as `.Rda` R dataframes (equivalent content). Each directory contains `MatlabSamplePlots.m` and rich R Markdown documentation (`.Rmd` + rendered `.html`).

### Shared `.mat` Data Structure

```
procedure    (string) — experimental procedure name
description  (string) — detailed procedure description
step[]  (one entry per current step)
├── comment       operation description ("charge", "discharge", "rest", "reference", "pulsed", …)
├── type          'C' (charge) | 'D' (discharge) | 'R' (rest)
├── relativeTime  s (time vector from step start)
├── time          s (absolute time from experiment start)
├── voltage       V
├── current       A
├── temperature   °C
└── date          "dd-Mon-yyyy HH:MM:SS" (step start timestamp)
```

Capacity is not pre-computed — it is derived by integrating current over reference discharge steps.

---

### RW Group A — Uniform Discharge, Room Temp (RW3–RW6)

- **Directory:** `Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/`
- Temperature: room temperature
- Charge: 2 A CC to 4.2 V, then CV until < 0.01 A (full charge)
- Discharge: random currents **0.5–4 A, uniform distribution, 5-minute steps**
- Reference cycles inserted every **50 RW cycles**: low-current OCV discharge (0.04 A), reference charge/discharge (2 A), pulsed load (1 A × 10 min, 20 min rest)
- File sizes: ~16–25 MB (.mat), ~9–14 MB (.Rda)

---

### RW Group B — Uniform Discharge + Variable Charge, Room Temp (RW1–RW2, RW7–RW8)

- **Directory:** `Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/`
- Temperature: room temperature
- Charge: **randomly selected duration** (0.5 h, 1 h, 1.5 h, 2 h, 2.5 h, or full), 2 A CC/CV
- Discharge: random 0.5–4 A, uniform, 5-minute steps
- Reference every **50 RW cycles**; pulsed charge profiles every **100 RW cycles**
- Contains RW1 and RW2 — the batteries cited in CLAUDE.md as the primary renewable-proxy set
- File sizes: ~21–27 MB (.mat), ~12–14 MB (.Rda)

---

### RW Group C — Uniform Charge & Discharge (RW9–RW12)

- **Directory:** `Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/`
- Temperature: room temperature
- **Unique:** both charge and discharge are randomized; current set:
  `{±0.75, ±1.5, ±2.25, ±3, ±3.75, ±4.5 A}` — negative = charging
- Operating window: **3.2–4.2 V**, 5-minute steps; bounds are hit increasingly often as battery degrades
- Reference cycles every **1500 RW steps** (~5 days); pulsed profiles every **3000 steps**
- Contains RW9 — cited in CLAUDE.md as the third renewable-proxy battery
- Largest files in the collection: ~82–86 MB (.mat), ~37–39 MB (.Rda)

---

### RW Group D — Skewed-High Discharge, Room Temp (RW17–RW20)

- **Directory:** `RW_Skewed_High_Room_Temp_DataSet_2Post/`
- Temperature: room temperature — currents: **0.5–5 A, 1-minute steps**
- Probability skewed toward **high currents**:
  `0.5A(2%) 1A(2.4%) 1.5A(3.6%) 2A(6%) 2.5A(9.2%) 3A(11.8%) 3.5A(17.2%) 4A(23.4%) 4.5A(19.4%) 5A(5%)`
- Reference cycles and pulsed profiles every **50 RW cycles**
- File sizes: ~16–17 MB (.mat), ~6.4–7.0 MB (.Rda)

---

### RW Group E — Skewed-High Discharge, 40 °C (RW25–RW28)

- **Directory:** `RW_Skewed_High_40C_DataSet_2Post/`
- Temperature: **40 °C** — same high-current skew as Group D
- Elevated temperature accelerates degradation → shorter lifespan, smaller files (~9–12 MB .mat)
- Good pair with Group D for isolating **temperature effect** at the same load distribution

---

### RW Group F — Skewed-Low Discharge, Room Temp (RW13–RW16)

- **Directory:** `RW_Skewed_Low_Room_Temp_DataSet_2Post/`
- Temperature: room temperature — currents: **0.5–5 A, 1-minute steps**
- Probability skewed toward **low currents**:
  `0.5A(7.2%) 1A(14.8%) 1.5A(19.3%) 2A(21.6%) 2.5A(14.6%) 3A(10%) 3.5A(6.5%) 4A(4%) 4.5A(1.5%) 5A(0.5%)`
- Reference cycles and pulsed profiles every **50 RW cycles**
- File sizes: ~18–24 MB (.mat), ~7.8–10 MB (.Rda)

---

### RW Group G — Skewed-Low Discharge, 40 °C (RW21–RW24)

- **Directory:** `RW_Skewed_Low_40C_DataSet_2Post/`
- Temperature: **40 °C** — same low-current skew as Group F
- Good pair with Group F for isolating **temperature effect** at the same load distribution
- File sizes: ~16–17 MB (.mat), ~7.1–7.5 MB (.Rda)

---

## Controlled vs. Randomized at a Glance

| Aspect | Controlled | Randomized |
|---|---|---|
| Discharge profile | Fixed current | Random walk (uniform or skewed) |
| Step duration | Full cycle (~hours) | 5 min or 1 min per step |
| Charge profile | Fixed CC/CV | Fixed (Groups A–D) or variable duration (Group B) |
| Impedance via EIS | Yes (dedicated cycles) | No — inferred from pulsed reference profiles |
| Capacity readout | Direct (2.7 V cycles only) | Computed from reference discharge integration |
| Temperatures | 4 °C, 24 °C, 43 °C | Room temp (~24 °C) and 40 °C |
| File formats | `.mat` only | `.mat` + `.Rda` |
| Documentation | Brief `.txt` | Rich `.Rmd` + `.html` with code examples |
| Batteries | B0005–B0056 (28 total) | RW1–RW28 (28 total) |

---

## Recommended Batteries for This Project

| Priority | Batteries | Reason |
|---|---|---|
| **Primary baseline** | B0005, B0006, B0007 | Canonical NASA aging dataset; cited in CLAUDE.md; room-temp controlled; longest runs |
| **Renewable-proxy** | RW1, RW2, RW9 | Cited in CLAUDE.md; simulate erratic grid-storage charge/discharge profiles |
| **Temperature analysis** | B0029–B0032 (hot) + B0041–B0044 (cold) | Isolate temperature effect against Group 1 baseline |
| **Extended randomized** | RW17–RW20 vs RW13–RW16 | High- vs low-skew load comparison at room temp |
| **Avoid** | B0049–B0052 | Truncated by software crash — unreliable EOL |
| **Skip (duplicate)** | Group 2 controlled | Same data as Group 3 B25–B28 |
