# Within-Cycle Data Prep Summary — `05_dataset_prep_dl.ipynb`

## What we did

Extracted the **CC charge-curve tensors** (V, I, T) preceding each retained discharge cycle from
the raw NASA `.mat` files, resampled them to a fixed-length representation, and joined them to the
SOH labels and LOBO splits from the processed CSVs.

## Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Voltage window | **3.8–4.1 V** | Sits fully in the CC phase (below the ~4.15 V CV taper).  Verified: >98% of controlled charges have >10 raw samples in the window; RW reference charges start higher (see below). |
| Resample length | **N = 128 pts** | Balances curve-shape detail vs model size.  RW upsampled from fewer raw pts (see caveat). |
| Channels | **V, I, T** (3-channel) | Current absorbs the CC charge-rate domain shift (controlled 1.5 A vs RW reference 2 A).  During a CC slice, I ≈ constant → a clean C-rate tag that lets the model disentangle rate from degradation. |
| Pairing rule | **Charge immediately preceding the retained discharge** | Controlled: raw-array adjacency, replicating `cycles_to_df` discharge-counter + capacity filter.  Randomized: preceding reference-charge step (`type='C'` + "reference" in comment), replicating the `rw_steps_to_df` quality gates + `iloc[::2]` decimation. |
| Label | **`capacity_retention`** (= capacity_ahr / baseline) | Non-leaky: input is a charge slice; label is from discharge-integration.  `energy_retention` deliberately excluded (≈ ∫V·I dt — integration leak). |

## Dataset stats

| | Count |
|---|---|
| Total samples (after join) | **778** |
| Controlled (B0005–B0007, B0018, every cycle) | 631 |
| Randomized (9 RW batteries, reference cycles only) | 147 |
| Dropped (window not covered) | 23 |
| **Train** | 567 |
| **Test** | 211 |

LOBO split (whole-battery, consistent with notebooks 03/04):

| Train | Test |
|---|---|
| B0005, B0006, B0018 | B0007 |
| RW1, RW9, RW13, RW14, RW16, RW17 | RW15, RW19, RW20 |

## Important caveat — RW voltage coverage

Randomized reference charges often start above 3.8 V (median V_min ≈ 3.90–3.97 V), so the raw
sample count inside the 3.8–4.1 V window is low (median 1–13 pts) before upsampling to 128.
Upsampling from very few points makes the RW tensors less informative than the controlled-battery
tensors, but they still contribute domain-shift diversity and 147 real labels.  The model will
encounter this as higher label noise for RW folds — consistent with what the macro-clock models
saw (RW sequences are inherently noisier).

## Output file

`data/processed/charge_curves.npz` — arrays:

| Key | Shape | Description |
|---|---|---|
| `X` | (778, 128, 3) float32 | Charge-curve tensors, channels = [V, I, T] |
| `soh` | (778,) float32 | `capacity_retention` — the SOH label |
| `cap_ahr` | (778,) float32 | Absolute capacity (Ahr) |
| `rul_frac` | (778,) float32 | Normalised RUL (for comparison to baselines) |
| `eol_index` | (778,) float32 | Per-battery EOL cycle |
| `battery_id` | (778,) object | Battery identifier |
| `cycle_index` | (778,) int32 | Cycle index (matches processed CSVs) |
| `split` | (778,) object | `'train'` or `'test'` |
| `scalars` | (778, 5) float32 | `[r_proxy, voltage_mean, temp_mean, temp_max, duration_s]` for nb06 RF baseline |

## What's next

**Notebook 06** — train 1D-CNN / TCN / LSTM on these tensors to estimate per-cycle SOH
(`capacity_retention`).  With ~800 samples and curve-shape information the baselines never
accessed, this is where DL is expected to earn its place.
