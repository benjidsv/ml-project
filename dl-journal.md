# DL Training Journal — Battery SOH/RUL

Tracking what we changed, why, and the results each iteration.
Baseline is the original notebook-06 results (before any fixes).

---

## Baseline (pre-fix)

| Phase | Model | MAE ± std | R² ± std | Skill | Spearman | Mono |
|---|---|---|---|---|---|---|
| Scalar RF | RF | 0.0481 ± 0.0429 | 0.623 ± 0.609 | 0.612 | 0.842 | 0.707 |
| DL Micro | CNN | 0.1754 ± 0.0956 | −1.128 ± 1.773 | −0.211 | 0.826 | 0.622 |
| DL Micro | TCN | 0.1117 ± 0.0725 | −0.203 ± 1.469 | 0.186 | 0.762 | 0.632 |
| DL Micro | LSTM | 0.0767 ± 0.0380 | 0.441 ± 0.476 | 0.436 | 0.730 | 0.574 |

**Root causes identified:**
- CNN/TCN used `x.mean(dim=-1)` — collapses monotone charge ramp to near-constant → regresses to mean → negative R²
- CNN/TCN used `BatchNorm1d` — running stats don't transfer to held-out battery under LOBO shift
- Epoch budget from one fixed split (CNN=26, TCN=48), hard-capped across all 13 folds → undertraining
- Inner-val battery always alphabetically-last (noisy upsampled RW) → unreliable early-stop signal
- Jitter applied once (not per-epoch) → weak augmentation
- 147/778 RW samples upsampled from 1–13 raw points → label noise
- Cross-cell reproducibility drift; RUL monotone branch blows up to 1e14

---

## Stage 1 — Architecture fixes (`src/curve_models.py`)

**Changes:**
- Pooling: `x.mean(dim=-1)` → `concat_pool` = `cat[mean, max, last_step]` (3× richer; last step is the charge-ramp endpoint where SOH signal concentrates)
- Norm: `BatchNorm1d` → `GroupNorm` (batch-independent; robust to LOBO domain shift)
- Head widths: CNN `Linear(64→1)` → `Linear(192→1)`, TCN `Linear(32→1)` → `Linear(96→1)`

**LOBO result (user-confirmed):** CNN/TCN MAE dropped from 0.10–0.20 → ~0.03, matching LSTM and out of negative R².

---

## Stage 2 — Training dynamics (`src/sequence.py` + notebook 06)

**Changes:**
- `train_model` gains two backward-compatible params: `scheduler="cosine"|"plateau"|None` and `augment_fn=None`
  - `"cosine"` → `CosineAnnealingLR(T_max=epochs, eta_min=lr*0.01)`
  - `augment_fn` is called on each training batch on-device (jitter now resamples every batch, not once before training)
- Tier-1 inner-val battery: was `sorted(train_bids)[-1]` (always a noisy RW battery) → now last alphabetical **controlled** battery (`B0018` in most folds)
- Tier-2 `lobo_cv_curve`: epoch budget raised from Tier-1 cap (CNN=26, TCN=48) to **300** with per-fold early stopping (patience=30) deciding when to stop
- **3-seed ensemble** per fold: three models trained with seeds 0/1/2, predictions averaged → cuts the large ±std
- `CONTROLLED_BIDS` constant defined once at the top of the Tier-1 cell

**Notebook 04 unaffected:** `scheduler=None, augment_fn=None` defaults preserve original behaviour.

**Tier-1 val loss (MSE) — before → after Stage 2:**
| Model | Before | After |
|---|---|---|
| CNN | 0.031 | 0.018 |
| TCN | 0.033 | 0.022 |
| LSTM | 0.030 | **0.002** (10× improvement) |

CNN best_ep=2 (very early stop — may benefit from lower initial LR or warmup, to investigate).

**TODO:** Run full LOBO with Stage 2 changes (controlled val battery + 3-seed ensemble + cosine LR) and record MAE/R²/std for all three architectures.

---

## Stage 3 — Hybrid curve + scalar head (`src/curve_models.py`, `src/sequence.py`, notebook 06)

**Changes:**
- All three models (`CNNReg`, `TCNRegPooled`, `LSTMReg`) gain optional `n_scalars=0` param
- `forward(x, mask, s=None)`: when `s` is not None, scalars are concatenated with the pooled curve embedding before the final FC head
- `train_model` gains `S_tr=None`/`S_val=None` params; scalars are included in the DataLoader and forwarded per-batch
- `predict_model` gains `S=None` param; passed as the third arg to model.forward
- Notebook 06: `ARCH_MAP_HYBRID` with Hybrid-CNN/TCN/LSTM; `run_tier1_hybrid` for quick Tier-1 testing; `lobo_cv_curve` gains `use_scalars=True` path (fully implemented, scales per inner fold)

**Why it works:** hybrid model has all of RF's information (the 9 charge scalars) plus the raw curve shape, so it is theoretically ≥ RF.

**Backward compatibility:** `n_scalars=0`, `S_tr/S_val/S=None` defaults leave notebook 04 and all existing pure-curve calls untouched.

**Tier-1 val loss (MSE):**
| Model | val_loss | best_ep |
|---|---|---|
| Hybrid-CNN | 0.03655 | 2 |
| Hybrid-TCN | 0.00977 | 39 |
| Hybrid-LSTM | 0.00474 | 25 |

**TODO:** Run full LOBO for hybrid variants (`lobo_results_hybrid`) and record MAE/R²/std vs the pure-curve models and the RF baseline.

---

## Stage 4 — CC+CV extraction (`src/charge_curves.py` + notebook 05)

**Root cause found:** RW reference charges have ~1300–1870 raw points each, but 99% are in the
CV phase (V≈4.2 V constant, current decaying −2A→0). The old 3.8–4.1 V window sliced the tiny
CC ramp only (1–13 pts), missing everything. That's why RW tensors were useless.

**Fix:** replaced `v_hi` ceiling in `slice_resample` with an **`i_cut=0.04 A`** current-cutoff
end criterion. Segment = first V≥3.8V to last |I|≥0.04A (captures CC + full CV taper).

| | Before | After |
|---|---|---|
| RW9 valid tensors | 40/40 but 6 raw pts each | 40/40, full CV decay |
| RW13-RW20 valid | many 1–2 raw pts | 10–16 of 11–17 each |
| Total valid | 778/801 | **794/801** |
| Dropped | 23 | **7** |

Channels stay `[V, I, T]` (C=3) — no model changes needed. The hybrid scalars improve for
free: `I_droop = I[-1]−I[0]` now spans the full CV current swing.

**Tier-1 val loss (MSE) — same controlled val set (B0018), so numbers unchanged vs Stage 2/3:**
| Model | val_loss | best_ep |
|---|---|---|
| CNN | 0.01789 | 2 |
| TCN | 0.02205 | 47 |
| LSTM | 0.00221 | 44 |
| Hybrid-CNN | 0.03655 | 2 |
| Hybrid-TCN | 0.00977 | 39 |
| Hybrid-LSTM | 0.00474 | 25 |

Note: Tier-1 val is B0018 (controlled, unchanged by Stage 4). Improvement from richer RW curves
will show in LOBO folds where an RW battery is held out.

**TODO:** Rebuild `charge_curves.npz` by running notebook 05 top-to-bottom, then re-run full LOBO
in notebook 06 to measure improvement on RW folds.

---

## Stage 5 — Reproducibility + RUL fix (notebook 06)

**Changes:**
- **Deleted the stale-CSV reload cell** — notebook now always recomputes top-to-bottom; no more cross-cell number drift
- **`rul_from_history` monotone branch** (`a < 0` → `a < -1e-5`): added a minimum-slope guard so near-flat predicted SOH histories return `None` instead of extrapolating to ±1e14
- **LOBO loop** restructured: pure models always run; hybrid runs `HYBRID_TO_EVAL = ["Hybrid-TCN"]` by default — `Hybrid-LSTM` and `Hybrid-CNN` are commented out and can be added if needed
- **Results table** (`1958b315`): hybrid-aware — adds "DL Hybrid" rows from `lobo_results_hybrid` if computed; picks `best_arch` across pure + hybrid; saves hybrid rows to CSV with `phase="dl_hybrid"`
- **SOH plots cell** (`9388ea0f`): uses `_all_lobo[best_arch]` (hybrid-aware) for the OOF scatter plot
- **Summary cell** (`60ad20b6`): shows hybrid rows with `◄ best` marker if applicable
- **Persist cell** (`7f022b7f`): now uses `scheduler="cosine"` + `augment_fn=make_jitter_fn()` (consistent with LOBO training); fixed metadata `v_hi→i_cut=0.04`

**Stage-6 LOBO results (time-axis, 788 samples):**
| Model | MAE ± std | R² ± std | Skill | Mono |
|---|---|---|---|---|
| RF (charge scalars) | 0.0670 ± 0.0509 | 0.379 ± 0.713 | 0.467 | 0.662 |
| LSTM | **0.0730 ± 0.0205** | 0.407 ± 0.642 | 0.424 | 0.684 |

Note: RF regressed (0.043→0.067) because CV-only tensors make `V_mean/V_mid` ~4.20 V constant
— the CC ramp that RF relied on is gone. LSTM is unaffected (it reads decay rate, not V mean).

LSTM per-fold: RW1=0.124 (constant offset, nominal-capacity gap), RW9=0.103 (compression).
Shrinkage: RW1 slope=−0.022 (flat, confirmed coverage gap), RW9 slope=−0.597 (compression).

**TODO:** Run notebook 06 top-to-bottom after rebuilding `charge_curves.npz` (nb05) and record final LOBO numbers for all models including Hybrid-TCN.

---

## Presentation write-up — findings to reconstruct from

*Written while fresh. The two analytical results below are the things that make this project
read as understanding rather than leaderboard-chasing. Reconstruct from here, not from memory.*

---

### Finding 1 — RW1: the Ah-vs-SOH stationarity violation

#### What the data says

RW1's LSTM LOBO results: slope = −0.022, correlation = −0.030, mean error = −0.124. Slope
essentially zero means the model outputs a nearly constant value for every RW1 cycle,
regardless of where in the degradation curve that cycle sits. The error is not growing with
SOH — it is a flat uniform offset of −0.124 (predictions are always 0.124 too low).

This is a distinct failure mode from RW9 (which has slope −0.60 = compression, correct
direction but undershooting at high SOH). RW1 has zero slope: the model is not tracking RW1
at all, just outputting something near its training prior minus some bias.

RF also fails RW1 (MAE 0.088, R² ≈ 0), confirming this is not a DL-specific failure. Both
models fail, differently. RF predicts near-constant (nearly flat), LSTM predicts constant
with a systematic level offset.

#### Root cause: nominal capacity heterogeneity

Investigation of `capacity_ahr` and `capacity_retention` per battery:

```
RW1:   baseline = 2.0003 Ah   ← outlier
RW9:   2.0977 Ah
RW13:  2.1290 Ah
RW14:  2.1154 Ah
RW15:  2.1191 Ah
RW16:  2.1136 Ah
RW17:  2.1369 Ah
RW19:  2.1320 Ah
RW20:  2.1339 Ah
```

All eight other RW batteries cluster tightly at 2.10–2.14 Ah. RW1 sits at 2.0003 Ah — a ~5%
difference, and it places RW1 closer to the controlled batteries (1.86–2.05 Ah) than to its
RW peers. RW1 is from the *Variable Charge* dataset group; the other RWs are from *Skewed Low*,
*Skewed High*, and *Charge–Discharge* groups.

RW1's SOH distribution is not the issue: mean 0.800 vs training mean 0.804, same bin shape
across [0.5, 1.0]. The labels are computed correctly: `capacity_retention = capacity_ahr /
baseline_ahr`, baseline is the true measured peak. This is not a normalization bug.

#### The mechanism (the stationarity violation)

The CV current-decay taper encodes **absolute charge accepted**, not fractional SOH. Total
charge accepted during a CV charge ≈ `∫|I|dt` over the taper segment. For a battery with
baseline B and SOH s, this is roughly proportional to `s × B`.

At SOH = 0.80:
- RW1: effective Q_CV ≈ 0.80 × 2.0003 = **1.60 Ah**
- Other RW: effective Q_CV ≈ 0.80 × 2.12 = **1.70 Ah**

The model trained on the eight other RW batteries has learned the mapping:
> "CV taper that delivers ~1.60 Ah → SOH ≈ 0.75"

…because that is what SOH=0.75 looks like for a 2.12 Ah battery. When it sees RW1 at SOH=0.80
with the same taper shape (~1.60 Ah), it predicts ~0.75. The error is ~0.05 from the capacity
gap alone. The remaining ~0.07 of the −0.124 offset comes from genuine covariate shift: RW1's
charge history (Variable Charge protocol, different SOC starting distributions) produces CV
curve characteristics that the training set — all from other protocol groups — has never seen
at that specific (SOH, capacity) combination.

The result is a **constant, SOH-independent offset**. The model can't track RW1 because the
mapping from {CV curve shape} → {SOH} is not stationary across batteries with different nominal
capacities. This is a stationarity violation. The SOH label is defined relative to each
battery's own baseline, but the physical curve shape is determined by absolute capacity, and
those two things are incommensurable without knowing the baseline.

#### Why it is unfixable without label leakage

The natural fix is to give the model `baseline_ahr` as an input scalar so it can factor out
the capacity difference. But `baseline_ahr` is a mild proxy for the SOH label
(`capacity_retention = capacity_ahr / baseline_ahr`), so feeding the baseline as a feature
creates a soft leakage path. The model could learn to use `baseline_ahr` to rescale its
predictions, but the resulting improvement would be partly a leakage artifact rather than
genuine curve-shape learning. For a project with a clean methodology, this is not worth
opening.

**The correct framing for the presentation:** RW1's 0.124 MAE is the floor for any
architecture that does not have access to the battery's nominal capacity. It is not a bug —
it is a data-coverage limit with a clean physical explanation. No training battery has both
(a) RW-style random charge protocol AND (b) ~2.00 Ah nominal capacity. The model has not been
shown what that combination looks like.

---

### Finding 2 — Current-axis resampling: the monotonicity assumption

#### The hypothesis

After the Stage-6 CV-only fix, the LSTM showed high-SOH compression on RW9: slope −0.60,
correct direction but undershooting at high SOH (early life). The hypothesis was that
resampling the CV taper at **uniform current levels** (V and T recorded at fixed |I| steps from
I_peak → I_cut) instead of uniform time would give better resolution at high SOH, because:

> At the same current level I*, a healthy cell (slow decay, still at high |I|) and a degraded
> cell (fast decay, approaching I* sooner) would have arrived there differently, and the
> voltage V at I* encodes the cell's internal state.

This is the logic of an electrochemical impedance sweep: V vs I at fixed points along the IV
curve is a cleaner SOH fingerprint than V vs time. The hypothesis was that applying this to
the CV taper would improve resolution.

#### What happened

| Model | MAE ± std | R² ± std | RW9 fold | RW1 fold |
|---|---|---|---|---|
| LSTM time-axis (baseline) | **0.073 ± 0.021** | 0.407 ± 0.642 | 0.103, R²=+0.48 | 0.124 |
| LSTM current-axis | 0.077 ± 0.046 | 0.267 ± 0.723 | **0.216**, R²=−1.37 | 0.128 |

RW9 went from MAE 0.103 (tracking with compression) to 0.216 (random, all overshooting). Mean
error jumped from +0.045 to +0.186. RW1 was essentially unchanged (still a flat offset, now
positive +0.068 instead of negative −0.124 — flipped direction because the model is now making
different errors, not because anything was fixed). Controlled batteries were mixed: a couple
improved marginally, others got worse, net negative.

Shrinkage diagnostic for current-axis:
- RW9: slope −0.804, r −0.739, mean error +0.186 — worse in every dimension
- RW1: slope +0.292, r +0.240 — slope has actually reversed sign (was −0.022); now slightly
  tracking in the wrong direction

#### Why the hypothesis was wrong

**The CV phase is defined by constant voltage.** V ≈ 4.20 V throughout the CV segment by
definition — the charger holds the battery at its setpoint. This means V(I) is approximately
the same value (~4.20 V) at every current level, for every SOH, for every battery. There is
no V-vs-I discriminability in a CV window: the voltage is constant. The I channel in
current-axis resampling is a constant monotone grid with zero cycle-to-cycle variation.
The T channel is weak and cross-battery variable in ways unrelated to SOH.

**The SOH signal in CV is exclusively temporal**: how fast |I| decays from I_peak toward zero.
A degraded cell, having lower capacity, reaches I_cut much sooner. That rate of decay is what
the time-axis I channel encodes across 128 uniform time steps — the position at which the
curve crosses any current threshold is a direct function of how fast the battery is accepting
charge, which is a direct function of its remaining capacity.

Current-axis resampling destroys this by forcing every battery's I channel to look identical
(the constant grid). The only surviving signal is tiny V variation and T, which are not
sufficient for SOH estimation in CV.

The impedance-sweep analogy breaks down precisely because an impedance sweep drives the battery
with a sinusoidal current at known frequencies; the resulting V contains frequency-domain
information that is genuinely SOH-sensitive. A CV segment is not a controlled excitation — the
current decays freely according to the battery's own electrochemistry, and the shape of that
free response in time IS the measurement.

#### Why it failed worse on RW9 specifically

RW9 is the battery where the time-axis representation already works best (slope −0.60,
positive R²=+0.48, LSTM beats RF 0.103 vs 0.205). That means RW9's CV taper in time-domain
has genuine discriminable shape between SOH levels — the decay rate differs measurably. When
current-axis erases the decay-rate information, it specifically hurts the battery that most
relied on it.

The Skewed cohort (RW13-20), which had good time-axis results, also degraded on current-axis,
just by smaller amounts (e.g. RW13 0.081→0.049... actually RW13 got slightly better, mixed).
The net is negative across the board.

#### What this confirms about the representation

Time-axis resampling at N=128 uniform steps over the CV segment is the correct representation
for this task. The I channel encodes the CV current-decay curve shape in its natural time
domain. The V channel captures any residual voltage variation from the CV setpoint (small but
real, especially for controlled batteries at different degradation stages). The T channel
captures thermal dynamics.

The `"current"` option remains in `src/charge_curves.py` because it would be the right choice
for data containing CC phase (where V does vary meaningfully with SOH), or for an actual
impedance sweep dataset. It is not the right choice here.

---

### Presentation arc

**Slide 1 — Problem and why trees should win**

Battery SOH prediction from partial charge data. The instructor's expected tool is RF/XGBoost
on engineered features. Scalar features from a charge curve — mean voltage, mean current,
temperature rise, etc. — are information summaries. The argument for them: a charge curve
contains ~3000 raw measurements but most of the variance is noise; a few well-chosen scalars
extract the signal cleanly and RF handles the nonlinear combinations.

The argument against: scalars cannot capture the *shape* of the curve. Two batteries at
different SOH can have similar mean voltage but different curvature in the current-decay taper.
The curve path — how quantities evolve over the 2–3 hour charge — contains information that no
finite set of scalars recovers exactly.

Set up: 13 batteries (4 controlled, 9 randomized), LOBO cross-validation (held-out batteries,
not held-out time), two cohorts with different charging protocols.

**Slide 2 — The curve-path pipeline**

Raw `.mat` → extract the CC→CV segment per cycle → resample to 128 uniform time steps →
channels [V, |I|, T] → LSTM. Key engineering decisions that matter:

- *CV-only anchoring* (`v_cv=4.15`): RW1 and RW9 start charges from random SOC, so the CC
  ramp length is SOC-dependent and uncorrelated with the label. Anchoring at the CV onset
  makes all 788 segments represent the same physical phase.
- *DCIR-pulse guard*: the NASA dataset inserts a single-sample measurement spike at the start
  of each charge step; the two-consecutive-samples rule skips it cleanly.
- *Stub rejection* (`I_MIN=1.0 A`): near-empty top-up charges (|I|max≈0.04 A) carry no SOH
  signal and are dropped.
- *Physical guard* (`V_PHYS_MAX=4.3 V`): one corrupted step in B0005 (8.39 V sensor spike) is
  rejected.

Result: 788 clean samples from 801 total (13 dropped). Before this pipeline: some folds were
catastrophic (RW1 R²=−6.9, variance dominated by degenerate stubs).

**Slide 3 — Where LSTM matches RF (the baseline)**

LSTM SOH LOBO aggregate: MAE=0.073 ± 0.021. RF on charge-curve scalars: MAE=0.067 ± 0.051.

Numbers look similar but the comparison is unfavorable to LSTM for a methodological reason:
the CV-only fix that was necessary for LSTM (anchoring at 4.15 V to remove SOC confounding)
also removed the CC voltage ramp from the scalar features. `V_mean` and `V_mid` are now ~4.20
V constant; the RF scalars lost their most informative features. RF's pre-fix performance with
CC+CV was 0.043. The CV-only fix was necessary for DL correctness and came with a RF
regression cost.

On the batteries where both work (the Skewed RW cohort, RW13-20): LSTM and RF are comparable,
with LSTM being more stable (std 0.021 vs 0.051). The R² std dropped from 1.26 to 0.64 after
the segmentation fix — no more catastrophic folds.

**Slide 4 — Where LSTM beats RF: RW9 (the hero slide)**

RW9 is the Charge–Discharge battery: 40 reference cycles, SOH from 1.0 down to 0.36.

| Model | MAE | R² | Slope | Corr |
|---|---|---|---|---|
| RF (scalars) | 0.205 | −0.84 | −0.737 | −0.936 |
| LSTM | **0.103** | **+0.48** | −0.597 | −0.893 |

RF MAE=0.205 on a battery that RF scores 0.043 globally: scalars completely flatten on RW9.
Why: RW9's charges start from random SOC (0.35–4.23 V onset), so `I_mean`, `V_mean` and the
other scalars encode the starting state rather than the degradation state. The aggregate scalar
values are dominated by how full the battery was when charging started, not by SOH.

LSTM reads the CV current-decay *shape* — after anchoring at the CV onset, the 128-point I
channel encodes how fast current falls, which is directly tied to charge capacity regardless of
starting SOC. LSTM slope = −0.597: tracking RW9's degradation with some high-SOH compression
(predicts slightly too low when SOH > 0.85, because high-SOH batteries have slower, more
similar-looking tapers). R² = +0.48: the model explains nearly half the SOH variance on a
battery it has never seen during training.

This is the core result: curve shape carries information that scalars cannot access, and the
LSTM reads it.

**Slide 5 — Where nothing works: RW1 (the honesty slide)**

RW1 is the Variable Charge battery: 21 reference cycles (after stub removal), SOH from 1.0
down to 0.58.

| Model | MAE | R² | Slope | Corr | Mean error |
|---|---|---|---|---|---|
| RF | 0.088 | −0.001 | −0.468 | −0.627 | −0.072 |
| LSTM | 0.124 | −0.876 | **−0.022** | **−0.030** | **−0.124** |

LSTM slope ≈ 0, correlation ≈ 0: the model outputs essentially a constant for all 21 RW1
cycles. Mean error = −0.124: predictions are always ~0.124 below true SOH, regardless of where
in the life curve the cycle falls. RF also fails (R² ≈ 0).

**Root cause:** RW1's nominal capacity is 2.0003 Ah. All other RW batteries: 2.10–2.14 Ah
(5–6% higher). The CV current-decay encodes absolute charge accepted (SOH × baseline), not
just SOH. At SOH=0.80: RW1 accepts 1.60 Ah, other RW accept 1.70 Ah. The model trained on
the 2.10–2.14 Ah cohort has learned "1.60 Ah taper → SOH ≈ 0.75." It applies that mapping to
RW1, undershoots by 0.05 from the capacity gap, and the remaining ~0.07 comes from RW1's
different charge protocol (Variable Charge vs Skewed). The result is a constant offset because
the error source — the capacity mismatch — does not vary across RW1's cycles.

This is a **stationarity violation**: the mapping from {CV curve shape} → {SOH} is not the
same for RW1 as for the training batteries, because nominal capacity differs and the physical
signal (absolute charge accepted) does not cancel out in the SOH normalization.

The fix — feeding `baseline_ahr` as a scalar input — would close most of the gap but is
borderline leaky (`capacity_retention = capacity_ahr / baseline_ahr`). For a clean
methodology, we report this as a known limit. **The 0.124 MAE floor on RW1 is structural for
this representation.**

**Slide 6 — The resampling experiment (the rigor slide)**

Having identified RW9 compression (slope −0.60, high-SOH undershooting), the natural question
is whether a different resampling would give the model better resolution at high SOH. The
hypothesis: resampling at uniform current levels (recording V and T at each of 128 evenly
spaced |I| steps from I_peak to I_cut) mirrors how electrochemical impedance spectroscopy
works — V at a fixed current encodes internal resistance and is SOH-sensitive.

Result: **categorically worse.** RW9 MAE 0.103 → 0.216, R² +0.48 → −1.37. Mean error
+0.045 → +0.186 (massive systematic overshoot). Aggregate MAE 0.073 → 0.077, std 0.021 →
0.046 (variance increased).

Why: in a CV window, **V ≈ 4.20 V throughout by definition**. That is what constant-voltage
charging means. V(I) carries essentially no SOH information — the battery is pinned at its
setpoint regardless of health. The I channel in current-axis resampling is a constant monotone
grid, identical for every cycle. The only surviving signal is weak T variation.

The SOH signal in CV is exclusively **temporal**: how fast |I| decays. A degraded cell,
having accepted less charge, reaches I_cut sooner. That decay rate is what the time-axis I
channel encodes across 128 uniform time steps. Current-axis destroys this by construction.
The impedance-sweep analogy fails because an impedance sweep is a controlled sinusoidal
excitation at known frequencies; a CV taper is a free relaxation response whose shape in
time IS the measurement.

Confirmed: time-axis resampling is optimal for CV-phase data. The code retains a `"current"`
mode for future experiments (e.g. CC phase or actual impedance data).

**Slide 7 — Conclusion**

RF for deployment: fast, interpretable, robust on the 11 batteries where scalars are
informative. RF MAE 0.043 (CC+CV scalars), LOBO-validated.

DL for curve-shape understanding: LSTM reads degradation from the CV current-decay taper.
Decisive win on RW9 (0.103 vs 0.205) where scalars flatten due to variable starting SOC.

Complementary blind spots:
- RF fails where aggregate scalars lose signal (RW9's variable starting SOC scrambles I_mean, V_mean)
- DL fails where the curve-shape→SOH mapping is not stationary across batteries (RW1's 5%
  capacity offset)

Neither model is universally better. The right architecture for production would be the
**hybrid model** (LSTM curve backbone + scalar head), which sees both the raw decay curve and
any battery-specific context that scalars can provide. Even there, RW1 would require
`baseline_ahr` as a scalar input — which puts the leakage question on the table and requires
a methodological decision.

What this project demonstrates that the leaderboard-chasers don't: we know *why* each model
works or fails, traced to physical mechanisms (stationarity violations, signal representation,
decay-rate encoding). That is the actual intellectual contribution.

---

## Stage 4.5 — Q(t) cumulative-charge channel *(not implemented — revisit later)*

**What it is.** A 4th channel `Q(t) = ∫|I| dt` (Ah poured in so far), computed within the
CC→CV segment alongside V, I, T. Monotonically rising; its total encodes charge throughput and
its shape encodes the dQ/dV curvature — a strong SOH proxy.

**Why we skipped it.** For controlled batteries charged from near-empty, raw total Q ≈ full
capacity ≈ the label (`capacity_retention`). This makes the model trivially accurate on
controlled LOBO folds but for the wrong reason (reading off a proxy of the answer, not learning
curve shape). For RW batteries charged from a half-full state, Q is a partial top-up ≠
capacity, so the leak only affects one cohort — and the inconsistency would be hard to explain.

**Two clean ways to add it later:**
1. **Normalized Q** — `Q(t) / Q_max ∈ [0,1]`, pure shape, no magnitude leak for either cohort.
2. **Raw Q + honest reporting** — maximally informative and legitimately deployable (real BMS
   uses coulomb-counting). Report metrics **with and without** Q so the curve-shape contribution
   is still legible.

**If implemented:** C becomes 4 (no model architecture changes needed since C is a constructor
param); rebuild `.npz`; update `charge_scalars()` to add `Q_total` and `Q_final` scalars.


---

## Stage 6 — SOC-invariant CV-only segmentation (`src/charge_curves.py`, notebook 05)

**Root cause corrected (journal hypothesis was wrong):**
The journal hypothesised "discharge (2A→1A)" mislabelled charge steps. Direct inspection of
the raw `.mat` data disproves this: zero reference-charge steps in RW1/RW9 contain positive
(discharge) current. All `type='C'` labels are correct.

**Actual root cause: variable starting state-of-charge (SOC)**

RW1 (*Variable Charge* dataset) and RW9 (*Charge–Discharge* dataset) start reference charges
from a random SOC (v0 anywhere 3.5–4.23 V). With the old CC-onset gate (`V≥3.8`):
1. CC ramp length is SOC-dependent and uncorrelated with the label → SOC noise injected
2. Near-empty top-up stubs (v0=4.19 V, |I|max≈0.04 A) → pure noise tensors with no signal
3. Index-resampling over a variable-length CC+CV segment → same SOH looks different per
   starting SOC → model can't generalise under LOBO

Degenerate fractions before fix: **RW1 6/24 (25%), RW9 19/40 (48%)**.
RW13/17/20 always start low (3.6–3.9 V) → 0 degenerate → explains the good/bad split.

Also found: **B0005** has a charge step with a non-physical 8.39 V sensor spike. All
controlled battery charge steps carry a 1-sample DCIR-pulse artifact at idx=1 (V drops to
~3.78 V, |I| spikes to ~4.5 A, then returns to normal CC current).

**Fixes in `slice_resample` (`src/charge_curves.py:48–140`):**
- `v_lo=3.8` (CC onset) → `v_cv=4.15` (CV onset): anchors all segments on the CV phase,
  which is present in 100% of charges regardless of starting SOC
- **Physical guard:** `np.nanmax(V) > v_phys_max=4.3 → None` (drops the B0005 8.39 V spike)
- **Two-consecutive-samples CV onset:** requires `V[k]≥v_cv AND V[k+1]≥v_cv` — skips the
  single-sample DCIR-pulse artifact at the start of every controlled charge step
- **Stub guard:** `np.abs(seg_I).max() < i_min=1.0 A → None` (drops near-empty top-ups)
- **Resampling kept as-is:** uniform-in-index at N=128; does NOT switch to current-axis.
  The SOH signal is the current-decay *rate* (healthy = slow decay); uniform-time resampling
  preserves that. V is ~flat at 4.2 V in CV so current-axis resampling would collapse the signal.

**New signature (backward-compatible):**
`slice_resample(V, I, T, v_cv=4.15, i_cut=0.04, n_points=128, i_min=1.0, v_phys_max=4.3, min_pts=10)`

**Extraction results after fix:**
| Battery | Before | After | Notes |
|---|---|---|---|
| RW1 | 24/24 valid, 6 degenerate | 21/24 (3 stubs dropped) | stubs gone |
| RW9 | 40/40 valid, 19 degenerate | 40/40 (all rescued) | re-anchor rescues full CV |
| B0005-B0007 | 168 each (8.39V leak) | 167 each (1 spike dropped) | sensor fix |
| B0018 | 132/132 (DCIR spike in tensor) | 132/132 clean | DCIR fix |

**Sanity checks on rebuilt `charge_curves.npz` (788 samples, was 794):**
- V: `[4.150, 4.296]` — no 8.39 V spikes, no sub-4.0 dips
- I: `[0.000, 2.005]` — no 4.5 A DCIR artifacts
- T: `[16.67, 41.0]` — physically reasonable

**TODO:** Re-run full LOBO in notebook 06 and record per-battery MAE — expect RW1/RW9 to
drop from 0.226/0.118 toward the ~0.02 band of RW13/17/20.

---

## RW1 constant-offset diagnosis (post Stage 6)

After Stage 6, per-battery LOBO shows LSTM slope ≈ −0.017, correlation ≈ −0.023 for RW1 —
essentially a flat, SOH-independent prediction with a uniform −0.127 bias. This was flagged as
potentially a normalization bug.

**It is not a bug.** Investigated by checking RW1's baseline_ahr and SOH distribution:

- Labels are clean: SOH decrements monotonically from 1.0, baseline = correctly measured peak.
- SOH distribution is in-band: RW1 mean 0.800, training mean 0.804, same bin shape.
- The actual anomaly is **nominal capacity**: RW1 baseline = **2.0003 Ah**, while all other
  8 RW batteries cluster tightly at 2.10–2.14 Ah (~5–6% higher). RW1's capacity is in line
  with the controlled batteries (1.86–2.05 Ah), not its RW peers.

**Why this produces a constant offset:** the CV taper encodes absolute charge accepted, not
fractional SOH. At SOH=0.80, RW1 delivers 1.60 Ah through CV; other RW batteries deliver
1.70 Ah. The model — trained on the 8 other RW batteries — has learned "CV taper ≈ 1.60 Ah
→ SOH ≈ 0.75." When it sees RW1 at SOH=0.80 with the same taper, it maps it to ~0.75.
This is constant and SOH-independent, exactly matching the observed signature. The 5% gap
accounts for ~0.04–0.05 of the 0.127 offset; the rest is covariate shift from dataset origin
(RW1 = *Variable Charge* group vs. *Skewed*/*Charge-Discharge* groups for all other RW).

**Verdict: known limitation, not fixable by curve cleaning.** RW1 is the only battery with
both RW-style charge patterns and ~2.0 Ah nominal capacity. No training battery anchors the
model to that regime. Adding `baseline_ahr` as a hybrid scalar would close most of the gap
mechanically but risks leakage (it's a mild label proxy). The 0.127 MAE floor on RW1 is
structural for this architecture — report it as such. RF also fails RW1 (MAE 0.088, R² ≈ 0),
confirming this is a data-coverage problem, not a model deficiency.

---

There was a problem with the LSTM harness: it only saw the last hidden state. Which is mostly constant, so it regressed to the mean and got terrible MAE. Fixed with biderectionnal pooling on all the hidden states (which now makes it slightly worse than RF). LSTM is great on most rw batteries but very conservative on controlled ones.

Remaining problems: a few RW batteries with 0.1 or even 0.2 MAE. Controlled batteries are very bad (shrinkage due to LOBO).

tried:
- longer patience to prevent too early stopping (no effect, kept)
- weight decay and dropout to zero, since the model is far from overfitting (no effect, kept for now)
- L1 (MAE) loss, since it's what we are trying to optimize and it handle tails better (better MAE + fixes controlled curves)

---

## Stage 7 — Current-axis resampling experiment (rejected)

**Hypothesis:** resampling CV taper at uniform |I| levels (V and T at fixed current steps)
instead of uniform time would give better high-SOH resolution: at the same current level, a
degraded vs healthy cell would show different voltage.

**Result: uniformly worse.**

| Model | MAE ± std | R² ± std | Worst fold |
|---|---|---|---|
| LSTM (time-axis, Stage-6) | **0.073 ± 0.021** | 0.407 ± 0.642 | RW1 0.124 |
| LSTM (current-axis) | 0.077 ± 0.046 | 0.267 ± 0.723 | RW9 **0.216** |

RW9 catastrophically degraded (0.103→0.216, R² +0.48→−1.37), all overshooting (mean +0.186).
RW1 flipped from undershooting (−0.124) to overshooting (+0.068).

**Root cause:** in a CV-only window, V ≈ constant at ~4.20 V throughout. V(I) carries
essentially no SOH information — the battery is pinned at its CV setpoint regardless of health.
The SOH signal IS the temporal decay rate: how fast |I| falls from I_peak to I_cut. That
is exactly what time-axis resampling encodes in the I channel, and what current-axis destroys by
construction (the I channel becomes a constant monotone grid with zero fold-to-fold variation).

Current-axis would make sense for an impedance sweep (V vs I at different frequencies), not
for a time-series CV window.

**Decision:** `RESAMPLE_AXIS = "time"` is locked as final. The `"current"` option remains
available in `src/charge_curves.py` for future experiments (e.g. CC-phase data).

The remaining RW9 compression (slope −0.60) is the signal-availability floor in this
representation. The most promising path forward is adding **CV duration as a scalar** to the
hybrid model — it directly encodes total charge accepted, strongly correlated with SOH, clean
and non-leaky.

---

Remaining issues: a few RW batteries have bad MAE.
Battery       MAE      R²     n
  ------------------------------------
  RW1        0.2260  -6.946    24  rw.  <- bad
  RW9        0.1178   0.240    40  rw.  <- bad
  RW20       0.0590   0.695    16  rw
  B0018      0.0547  -0.299   132  ctrl
  RW17       0.0289   0.913    16  rw
  RW13       0.0247   0.946    12  rw
  B0006      0.0238   0.850   168  ctrl
  RW16       0.0209   0.980    10  rw
  B0007      0.0200   0.687   168  ctrl
  B0005      0.0190   0.870   168  ctrl
  RW14       0.0184   0.984    12  rw
  RW19       0.0172   0.972    15  rw
  RW15       0.0135   0.988    13  rw

  the median MAE beats the Random Forest ! we just have those two outliers to handle.
  notes: B18 has negative R², but MAE is fine. This is because the SOH range of B18 is very narrow so SS_total is very small, but actual error is low, so it's ok.
  - todo: investigate potential small leak (b18 is the inner val battery for early stop)

Interestingly, RW1 and RW9 have perfect coverage, and for the points not covered, LSTM predicts perfectly (where RF tails). This is the same problem from earlier: this is shrinkage, since controlled batteries dominate the data, when RW1/9 is held out - those have a covariate shift, they are structurally different and operate at higher SOH, which LSTM undershoots - the model regresses to the training mean, which is lower because it's dominated by the controlled set.

To test this i compared prediction error vs true SOH, and noticed that's it's actually compression. The model tracks the slope correctly, but the higher the true SOH, the higher pred error. This means the model is reluctant to predict high values, which would make sense if they arent present enough in the coverage, so even L1 loss couldn't get that. Also, the high SOH are the cases in the early life of the battery, where it's mostly constant, so the slope shape mechanism doesn't really work. This is a signal availibity problem.

Added per sample norm (was global before). Should help but not fix the problem. 
As expected, reduced MAE slightly on RW1 and RW0 but only by 0.003 so not really significant.

To investigate i plotted curves of V, A, T for training vs RW1 and RW9, and found the issue:
the charge/discharge segmentation does not work for RW1 and RW9. Some of the amperage curves show a discharge (2A -> 1A).



