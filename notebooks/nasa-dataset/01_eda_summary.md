# EDA Summary — `01_eda.ipynb`

## Datasets Loaded

**Controlled (lab, fixed-current discharge):**
- Room temp (24°C): B0005, B0006, B0007, B0018 — 132–168 cycles each
- Hot (43°C): B0029–B0032 — ~40 cycles each
- Cold (4°C): B0042–B0044 — 30–75 cycles each

**Randomized (renewable-proxy, random-walk load):**
- Uniform random: RW1, RW2, RW9
- Skewed load: RW13–RW20

## Key Data-Cleaning Decisions

| Decision | Rationale |
|---|---|
| Capacity baseline = max over first 10% of cycles | Li-ion at 43°C shows initial rise before fade; avoids retention > 1.0 |
| Drop cycles with capacity < 50% baseline | NASA-documented anomalous short discharges in B0041–B0044 |
| Cold group (4°C): keep only cycles with avg current ≥ 1.8A | B0042–B0044 interleave 1A and 4A protocols; Peukert effect makes mixed rates incomparable |
| B0041 excluded entirely | Only 25 cycles (vs 66–100 peers), anomalous starting capacity 1.33 Ahr |
| RW reference discharges: remove OCV sweep (≤ 0.5A mean) and pulsed sub-steps (< 1200s) | Removes non-reference partial/recovery discharges that form sawtooth pattern |

## Capacity Fade by Condition

| Condition | Starting capacity | End capacity | Fade |
|---|---|---|---|
| Room temp (B0005–B0018) | ~1.86 Ahr | 1.2–1.4 Ahr | 28–35% |
| Hot 43°C (B0029–B0032) | ~1.70 Ahr | 1.59–1.68 Ahr | 1–6% (never reaches EOL) |
| Cold 4°C (B0042–B0044) | varies | 0.889–1.49 Ahr | 10–50% (inconsistent) |
| Randomized (RW1–RW20) | 2.0–2.1 Ahr | 0.7–1.2 Ahr | 40–65% |

## Features Validated

- **Discharge energy (Wh):** V·|I| integration — captures both capacity loss and voltage sag from rising resistance
- **Resistance proxy (r_proxy):** Onset ΔVΔI = (V_rest − V_loaded) / |I_load|; field-deployable without EIS
- **EIS validation:** r_proxy vs true Rct from EIS gives r ≈ 0.77 (B0005: 0.78) — strong enough for deployment

## Plots Generated (`results/`)
- `controlled_capacity_fade.png` — Absolute and normalised fade by temperature
- `temperature_effect.png` — Mean retention per temperature (decile binning)
- `rw_capacity_fade.png` — Randomized capacity fade by load condition
- `lifespan_comparison.png` — Total cycles to EOL, all batteries
- `correlation_heatmap.png`, `rw_correlation_heatmap.png` — Feature correlations
- `proxy_vs_eis.png` — Onset proxy vs true Rct scatter (r ≈ 0.77)
