"""Voltage-aligned charge-curve extraction and LOBO evaluation harness.

Implements the SOTA input representation from the Attention-CNN-GRU /
GRU-CNN literature: sample |I|, T, and Δt at a fixed descending voltage
grid during the CC charging phase.

Why a descending voltage grid?
--------------------------------
Indexing 4.2 V → 3.8 V puts valid data at the START of the sequence and
any low-voltage truncation at the TAIL.  Tail-padding is the standard
convention for pack_padded_sequence, so the bidirectional LSTM never
ingests leading pad — no alignment acrobatics required.

RW1/RW9 start their reference charges from a random state-of-charge, so
some cycles may not reach V_LO = 3.8 V from below.  Those grid points
(at the tail) are masked out rather than dropped, keeping every RW cycle
in the dataset.

Target: Ct / C_nominal  (fractional rated capacity, C_nominal = 2.0 Ah)
This avoids the per-battery measured-baseline denominator used by the
earlier capacity_retention feature and supports adding future datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.integrate import cumulative_trapezoid
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

C_NOMINAL: float = 2.0   # NASA 18650 rated capacity (Ah)
V_LO: float = 3.8        # lower bound of the CC-window (V)
V_HI: float = 4.2        # upper bound = CV knee (V)
N_GRID: int = 128         # grid resolution

# .mat paths relative to project root — mirrors charge_curves.CONTROLLED_MATS
CONTROLLED_MATS: dict[str, str] = {
    "B0005": "data/controlled/1. BatteryAgingARC-FY08Q4/B0005.mat",
    "B0006": "data/controlled/1. BatteryAgingARC-FY08Q4/B0006.mat",
    "B0007": "data/controlled/1. BatteryAgingARC-FY08Q4/B0007.mat",
    "B0018": "data/controlled/1. BatteryAgingARC-FY08Q4/B0018.mat",
}

RANDOMIZED_MATS: dict[str, str] = {
    "RW1":  "data/randomized/Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW1.mat",
    "RW9":  "data/randomized/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW9.mat",
    "RW13": "data/randomized/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW13.mat",
    "RW14": "data/randomized/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW14.mat",
    "RW15": "data/randomized/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW15.mat",
    "RW16": "data/randomized/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW16.mat",
    "RW17": "data/randomized/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW17.mat",
    "RW19": "data/randomized/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW19.mat",
    "RW20": "data/randomized/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW20.mat",
}

CONTROLLED_BIDS: frozenset[str] = frozenset(CONTROLLED_MATS)


# ---------------------------------------------------------------------------
# Core voltage-grid resampling
# ---------------------------------------------------------------------------


def resample_voltage_grid(
    V: np.ndarray,
    I: np.ndarray,
    T: np.ndarray,
    Time: np.ndarray,
    v_lo: float = V_LO,
    v_hi: float = V_HI,
    n_grid: int = N_GRID,
    i_min: float = 1.0,
    v_phys_max: float = 4.3,
    min_pts: int = 10,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Extract a voltage-aligned charge-curve tensor for one charge step.

    Isolates the CC charging segment (charge onset → CV knee at v_hi),
    enforces a monotone-rising voltage axis, then interpolates |I|, T, and
    Δt at a fixed descending voltage grid vg = linspace(v_hi, v_lo, n_grid).

    Parameters
    ----------
    V, I, T, Time : 1-D arrays
        Voltage (V), current (A, either sign convention), temperature (°C),
        and time (s from step start or absolute) for a single charge step.
    v_lo, v_hi : float
        Voltage window bounds; default 3.8–4.2 V.
    n_grid : int
        Number of fixed grid points; default 128.
    i_min : float
        Minimum peak |I| in the CC segment; rejects stub charges.
    v_phys_max : float
        Physical voltage ceiling; any sample above this rejects the step.
    min_pts : int
        Minimum number of valid (covered) grid points before the step is
        dropped entirely.

    Returns
    -------
    tensor : (n_grid, 5) float32 or None
        Channels: [|I|, ΔT, t_elapsed_s, Q_Ah, dQ/dV].
          |I|       — absolute current in amps (C-rate normalisation done downstream).
          ΔT        — temperature rise relative to the CC-segment start (°C); removes
                      protocol/ambient offset that fingerprints cell pools with
                      constant-temperature imputation (e.g. CALCE ΔT ≡ 0).
          t_elapsed — seconds from CC-segment start.
          Q_Ah      — cumulative charge delivered (Ah); C-nom normalisation downstream.
          dQ/dV     — incremental capacity (Ah/V); computed over the descending grid.
        Uncovered grid positions filled with 0.0 (matches the scaled channel mean,
        the least-informative fill; re-zeroed by _reapply_fill after global_scale).
    mask : (n_grid,) bool or None
        True = valid grid position (voltage covered by this cycle).
        The valid region is a LEADING block (high-V end) because the grid
        descends from v_hi; truncation at the tail (low-V) is the standard
        right-padding convention compatible with pack_padded_sequence.
    cc_scalars : dict
        {"cc_dt": float, "cc_slope": float, "coverage": float}
        cc_dt     = seconds from v_lo crossing to v_hi knee (CC duration).
        cc_slope  = (v_hi - v_start) / cc_dt  (V/s), 0.0 if no ramp captured.
        coverage  = mask.mean().
    """
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    T = np.asarray(T, dtype=float)
    Time = np.asarray(Time, dtype=float)
    _empty: dict = {"cc_dt": 0.0, "cc_slope": 0.0, "coverage": 0.0}

    # Physical sanity guard (e.g. 8.39 V spike in B0005)
    if np.nanmax(V) > v_phys_max:
        return None, None, _empty

    abs_I = np.abs(I)

    # --- Locate CC charging segment ---
    # Start: first sample where |I| > i_min (charge onset, skip DCIR-pulse artifact)
    onset_cands = np.where(abs_I > i_min)[0]
    if len(onset_cands) == 0:
        return None, None, _empty
    i_onset = int(onset_cands[0])

    # End: CV knee — first index (after onset) where V >= v_hi with two consecutive
    # samples above threshold (avoids false triggers on noise spikes).
    above_hi = np.where(V[i_onset:] >= v_hi)[0]
    i_knee = None
    for k in above_hi:
        idx = i_onset + int(k)
        if idx + 1 < len(V) and V[idx + 1] >= v_hi:
            i_knee = idx
            break
    if i_knee is None:
        # Voltage never reached v_hi — cycle is too short / not fully charged.
        # Still try to use what's available up to the max voltage reached.
        if V[i_onset:].max() < v_lo:
            return None, None, _empty  # entirely below window
        i_knee = int(i_onset + np.argmax(V[i_onset:]))

    seg_V = V[i_onset: i_knee + 1]
    seg_I = abs_I[i_onset: i_knee + 1]
    seg_T = T[i_onset: i_knee + 1]
    seg_time = Time[i_onset: i_knee + 1]

    if len(seg_V) < 2 or seg_I.max() < i_min:
        return None, None, _empty

    # --- Compute cc_scalars ---
    t_elapsed = seg_time - seg_time[0]

    # Find v_lo crossing for cc_dt (may be missing for high-SOC starts)
    above_lo = np.where(seg_V >= v_lo)[0]
    if len(above_lo) > 0:
        t_lo = float(seg_time[above_lo[0]])
        t_hi = float(seg_time[-1])
        cc_dt = max(0.0, t_hi - t_lo)
        v_at_lo = float(seg_V[above_lo[0]])
        cc_slope = (v_hi - v_at_lo) / cc_dt if cc_dt > 0 else 0.0
    else:
        cc_dt = 0.0
        cc_slope = 0.0

    # --- Enforce monotone-rising voltage for np.interp ---
    # During CC charging V is near-monotone but raw data is noisy.
    # np.maximum.accumulate ensures the x-axis for interp is non-decreasing.
    seg_V_mono = np.maximum.accumulate(seg_V)

    # --- Build descending voltage grid ---
    vg = np.linspace(v_hi, v_lo, n_grid)  # (n_grid,) descending

    # --- Mask: True where the grid voltage is within the cycle's covered range ---
    v_min_covered = float(seg_V.min())
    v_max_covered = float(seg_V.max())
    mask = (vg >= v_min_covered) & (vg <= v_max_covered)

    covered = int(mask.sum())
    if covered < min_pts:
        return None, None, {**_empty, "coverage": covered / n_grid}

    # --- Cumulative charge Q(t) integrated over the CC segment (Ah) ---
    # cumulative_trapezoid returns n-1 values; prepend 0 for the segment start.
    Q_t = np.zeros(len(seg_I), dtype=np.float64)
    if len(seg_I) >= 2:
        Q_t[1:] = cumulative_trapezoid(seg_I, seg_time) / 3600.0

    # Temperature relative to segment start — removes protocol/ambient-offset bias.
    # For cells with imputed constant temperature (CALCE), ΔT ≡ 0 everywhere,
    # so it carries no dataset-identity information after global_scale.
    T_start = float(seg_T[0])

    # --- Interpolate all channels at covered grid voltages ---
    # seg_V_mono is non-decreasing; vg[mask] is the query in the same direction
    # (descending, but interp accepts out-of-order query). We query against the
    # ascending axis using vg_covered directly.
    vg_covered = vg[mask]  # still descending subset; np.interp requires xp ascending
    # Flip so xp is ascending for np.interp
    xp = seg_V_mono
    r_I_c   = np.interp(vg_covered, xp, seg_I)
    r_dT_c  = np.interp(vg_covered, xp, seg_T) - T_start  # ΔT from segment start
    r_te_c  = np.interp(vg_covered, xp, t_elapsed)
    Q_grd_c = np.interp(vg_covered, xp, Q_t)

    # dQ/dV: both Q and V decrease together on the descending grid, so the ratio
    # is positive.  np.gradient handles non-uniform and negative spacing correctly.
    if len(vg_covered) > 1:
        dQdV_c = np.gradient(Q_grd_c, vg_covered)
    else:
        dQdV_c = np.zeros_like(Q_grd_c)

    # Build full tensor, fill uncovered (tail) with 0.0.
    # _reapply_fill (called after global_scale) re-zeros these positions in the
    # scaled domain, so the exact raw fill value does not matter.
    r_I    = np.zeros(n_grid, dtype=np.float32)
    r_dT   = np.zeros(n_grid, dtype=np.float32)
    r_te   = np.zeros(n_grid, dtype=np.float32)
    r_Q    = np.zeros(n_grid, dtype=np.float32)
    r_dQdV = np.zeros(n_grid, dtype=np.float32)
    r_I[mask]    = r_I_c.astype(np.float32)
    r_dT[mask]   = r_dT_c.astype(np.float32)
    r_te[mask]   = r_te_c.astype(np.float32)
    r_Q[mask]    = Q_grd_c.astype(np.float32)
    r_dQdV[mask] = dQdV_c.astype(np.float32)

    tensor = np.stack([r_I, r_dT, r_te, r_Q, r_dQdV], axis=-1)  # (n_grid, 5) float32
    coverage = float(covered / n_grid)

    return tensor, mask, {"cc_dt": cc_dt, "cc_slope": cc_slope, "coverage": coverage}


# ---------------------------------------------------------------------------
# Controlled battery extraction
# ---------------------------------------------------------------------------


def extract_controlled_vg(
    cycles: list[dict],
    battery_id: str,
    v_lo: float = V_LO,
    v_hi: float = V_HI,
    n_grid: int = N_GRID,
    i_min: float = 1.0,
    v_phys_max: float = 4.3,
    min_pts: int = 10,
) -> list[dict]:
    """Extract voltage-grid tensors for one controlled battery.

    Mirrors cycles_to_df (battery.py): only discharge cycles are kept; the
    same capacity > 0.5*baseline filter and cycle_index re-indexing are
    applied so (battery_id, cycle_index) joins the processed CSVs exactly.

    Charge data fields: Voltage_measured / Current_measured /
    Temperature_measured / Time (all positive-current convention).

    Returns list of dicts with keys:
        battery_id, cycle_index, tensor (n_grid,3)|None, mask (n_grid,)|None,
        coverage, cc_dt, cc_slope.
    """
    pairs: list[tuple[dict, dict | None]] = []
    last_charge: dict | None = None

    for cycle in cycles:
        ctype = str(cycle.get("type", ""))
        if ctype == "charge":
            last_charge = cycle
        elif ctype == "discharge":
            pairs.append((cycle, last_charge))

    if not pairs:
        return []

    # Capacity filter (mirrors battery.py:85-89)
    cap_vals: list[float] = []
    for dis, _ in pairs:
        d = dis["data"]
        time = np.asarray(d["Time"], dtype=float)
        current = np.asarray(d["Current_measured"], dtype=float)
        cap_vals.append(float(np.trapezoid(np.abs(current), time) / 3600))

    n_warmup = max(5, int(0.10 * len(pairs)))
    baseline = max(cap_vals[:n_warmup])
    kept = [(dis, chg) for (dis, chg), cap in zip(pairs, cap_vals)
            if cap > 0.5 * baseline]

    if not kept:
        return []

    records: list[dict] = []
    for new_idx, (_, chg) in enumerate(kept):
        tensor, mask, cc_sc = None, None, {"cc_dt": 0.0, "cc_slope": 0.0, "coverage": 0.0}
        if chg is not None:
            d = chg["data"]
            V    = np.asarray(d["Voltage_measured"],    dtype=float)
            I    = np.asarray(d["Current_measured"],    dtype=float)
            Temp = np.asarray(d["Temperature_measured"], dtype=float)
            TT   = np.asarray(d["Time"],                dtype=float)
            tensor, mask, cc_sc = resample_voltage_grid(
                V, I, Temp, TT, v_lo=v_lo, v_hi=v_hi, n_grid=n_grid,
                i_min=i_min, v_phys_max=v_phys_max, min_pts=min_pts,
            )
        records.append({
            "battery_id":  battery_id,
            "cycle_index": new_idx,
            "tensor":      tensor,
            "mask":        mask,
            **cc_sc,
        })

    return records


# ---------------------------------------------------------------------------
# Randomized battery extraction
# ---------------------------------------------------------------------------


def extract_randomized_vg(
    steps: list[dict],
    battery_id: str,
    v_lo: float = V_LO,
    v_hi: float = V_HI,
    n_grid: int = N_GRID,
    i_min: float = 1.0,
    v_phys_max: float = 4.3,
    min_pts: int = 10,
) -> list[dict]:
    """Extract voltage-grid tensors for one randomized battery.

    Mirrors rw_steps_to_df (battery.py): keeps only reference-discharge
    steps passing quality gates, applies iloc[::2] decimation, and resets
    cycle_index = range(n).

    Charge data fields: voltage / current (negative convention → abs) /
    temperature / relativeTime.

    Returns list of dicts with keys:
        battery_id, cycle_index, tensor (n_grid,3)|None, mask (n_grid,)|None,
        coverage, cc_dt, cc_slope.
    """
    raw_pairs: list[tuple[dict, dict | None]] = []
    last_ref_charge: dict | None = None

    for step in steps:
        stype   = str(step.get("type", "")).upper()
        comment = str(step.get("comment", "")).lower()
        is_ref  = "reference" in comment

        if stype == "C" and is_ref:
            last_ref_charge = step
        elif stype == "D" and is_ref:
            time    = np.asarray(step["relativeTime"], dtype=float)
            current = np.asarray(step["current"],      dtype=float)
            if len(time) < 2:
                continue
            if np.abs(current).mean() < 0.5:
                continue
            if float(time[-1] - time[0]) < 1200:
                continue
            raw_pairs.append((step, last_ref_charge))

    if not raw_pairs:
        return []

    decimated = raw_pairs[::2]  # matches rw_steps_to_df line 170

    records: list[dict] = []
    for new_idx, (_, chg) in enumerate(decimated):
        tensor, mask, cc_sc = None, None, {"cc_dt": 0.0, "cc_slope": 0.0, "coverage": 0.0}
        if chg is not None:
            V    = np.asarray(chg["voltage"],      dtype=float)
            I    = np.asarray(chg["current"],      dtype=float)  # negative → abs inside
            Temp = np.asarray(chg["temperature"],  dtype=float)
            TT   = np.asarray(chg["relativeTime"], dtype=float)
            tensor, mask, cc_sc = resample_voltage_grid(
                V, I, Temp, TT, v_lo=v_lo, v_hi=v_hi, n_grid=n_grid,
                i_min=i_min, v_phys_max=v_phys_max, min_pts=min_pts,
            )
        records.append({
            "battery_id":  battery_id,
            "cycle_index": new_idx,
            "tensor":      tensor,
            "mask":        mask,
            **cc_sc,
        })

    return records


# ---------------------------------------------------------------------------
# Full dataset builder
# ---------------------------------------------------------------------------


def build_voltage_grid_dataset(
    project_root: Path,
    v_lo: float = V_LO,
    v_hi: float = V_HI,
    n_grid: int = N_GRID,
    i_min: float = 1.0,
    v_phys_max: float = 4.3,
    min_pts: int = 10,
) -> pd.DataFrame:
    """Extract voltage-grid tensors for all 13 retained batteries.

    Parameters
    ----------
    project_root : Path
        Repository root (parent of data/, src/).

    Returns
    -------
    DataFrame with columns:
        battery_id, cycle_index, tensor (ndarray|None), mask (ndarray|None),
        coverage, cc_dt, cc_slope.
    One row per retained discharge cycle, same cycle_index as the CSVs.
    """
    # Import loaders here to avoid a hard dependency at module level
    # when running from the project root where src/ is on sys.path.
    from battery import load_controlled, load_randomized

    all_records: list[dict] = []

    for bid, rel_path in CONTROLLED_MATS.items():
        path = project_root / rel_path
        cycles = load_controlled(path)
        records = extract_controlled_vg(
            cycles, bid, v_lo=v_lo, v_hi=v_hi, n_grid=n_grid,
            i_min=i_min, v_phys_max=v_phys_max, min_pts=min_pts,
        )
        valid = sum(1 for r in records if r["tensor"] is not None)
        mean_cov = np.mean([r["coverage"] for r in records]) if records else 0.0
        print(f"  {bid}: {len(records)} cycles, {valid} valid, mean coverage {mean_cov:.3f}")
        all_records.extend(records)

    for bid, rel_path in RANDOMIZED_MATS.items():
        path = project_root / rel_path
        steps = load_randomized(path)
        records = extract_randomized_vg(
            steps, bid, v_lo=v_lo, v_hi=v_hi, n_grid=n_grid,
            i_min=i_min, v_phys_max=v_phys_max, min_pts=min_pts,
        )
        valid = sum(1 for r in records if r["tensor"] is not None)
        mean_cov = np.mean([r["coverage"] for r in records]) if records else 0.0
        print(f"  {bid}: {len(records)} cycles, {valid} valid, mean coverage {mean_cov:.3f}")
        all_records.extend(records)

    return pd.DataFrame(all_records)


# ---------------------------------------------------------------------------
# Target attachment and array preparation
# ---------------------------------------------------------------------------


def attach_targets(
    curve_df: pd.DataFrame,
    project_root: Path,
    c_nominal: float = C_NOMINAL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Join curve tensors to processed-CSV labels; compute fractional-rated target.

    Inner-joins curve_df on (battery_id, cycle_index) to the processed CSVs,
    drops rows with missing tensors or labels, and stacks arrays.

    Returns
    -------
    X      : (n, n_grid, 3)  float32  — voltage-grid curve tensors
    mask   : (n, n_grid)     bool     — valid grid positions
    y      : (n,)            float32  — capacity_ahr / c_nominal
    groups : (n,)            object   — battery_id strings
    cidx   : (n,)            int64    — cycle_index
    """
    ctrl_csv = pd.read_csv(project_root / "data/processed/controlled.csv",
                           usecols=["battery_id", "cycle_index", "capacity_ahr"])
    rw_csv   = pd.read_csv(project_root / "data/processed/randomized.csv",
                           usecols=["battery_id", "cycle_index", "capacity_ahr"])
    labels   = pd.concat([ctrl_csv, rw_csv], ignore_index=True)

    merged = curve_df.merge(labels, on=["battery_id", "cycle_index"], how="inner")

    # Drop cycles where extraction failed (no charge preceding discharge, etc.)
    merged = merged[merged["tensor"].apply(lambda t: t is not None)].copy()
    merged = merged[merged["capacity_ahr"].notna()].copy()
    merged = merged.reset_index(drop=True)

    n = len(merged)
    n_grid = merged["tensor"].iloc[0].shape[0]

    X    = np.stack(merged["tensor"].values).astype(np.float32)   # (n, n_grid, 3)
    mask = np.stack(merged["mask"].values)                         # (n, n_grid) bool
    y    = (merged["capacity_ahr"].to_numpy(dtype=np.float32) / c_nominal)
    groups = merged["battery_id"].to_numpy()
    cidx   = merged["cycle_index"].to_numpy(dtype=np.int64)

    print(f"  Dataset: {n} samples, {n_grid} grid points, "
          f"y ∈ [{y.min():.3f}, {y.max():.3f}]")
    return X, mask, y, groups, cidx


# ---------------------------------------------------------------------------
# NPZ cache
# ---------------------------------------------------------------------------


def save_npz(path: Path, X, mask, y, groups, cidx,
             v_lo=V_LO, v_hi=V_HI, n_grid=N_GRID) -> None:
    """Save the extracted dataset to a .npz file."""
    np.savez_compressed(
        path,
        X=X, mask=mask, y=y, groups=groups, cidx=cidx,
        config=np.array([v_lo, v_hi, n_grid, C_NOMINAL]),
    )
    print(f"  Saved to {path}")


def load_npz(path: Path) -> tuple:
    """Load a previously saved voltage-grid dataset."""
    d = np.load(path, allow_pickle=True)
    return d["X"], d["mask"], d["y"], d["groups"], d["cidx"]


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def global_scale(
    X_tr: np.ndarray,
    X_val: np.ndarray,
    X_te: np.ndarray,
    mask_tr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel StandardScaler fit on valid train positions only.

    Fitting on masked-valid rows prevents the 0-fill of uncovered grid points
    from contaminating the channel mean/std (a bug in the nb08 approach).
    After transform, uncovered positions are re-zeroed so they represent the
    scaled channel mean (the least-informative fill for the LSTM).

    Parameters
    ----------
    X_tr, X_val, X_te : (n, L, C) float32
    mask_tr            : (n, L) bool — valid positions in the train split

    Returns
    -------
    X_tr_s, X_val_s, X_te_s — scaled arrays, same shapes as inputs.
    """
    n, L, C = X_tr.shape
    # Gather only valid (unmasked) rows: shape (n_valid, C)
    valid_rows = X_tr[mask_tr]          # (n_valid, C)

    scaler = StandardScaler()
    scaler.fit(valid_rows)

    X_tr_s  = scaler.transform(X_tr.reshape(-1, C)).reshape(n, L, C).astype(np.float32)
    nv, Lv, _ = X_val.shape
    X_val_s = scaler.transform(X_val.reshape(-1, C)).reshape(nv, Lv, C).astype(np.float32)
    nte, Lte, _ = X_te.shape
    X_te_s  = scaler.transform(X_te.reshape(-1, C)).reshape(nte, Lte, C).astype(np.float32)

    return X_tr_s, X_val_s, X_te_s


def _reapply_fill(X_scaled: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero-fill uncovered grid positions after scaling.

    After StandardScaler, the pre-fill 0.0 values become (0 - mean)/std ≠ 0.
    This function restores the neutral 0.0 fill for positions where mask=False.
    """
    out = X_scaled.copy()
    out[~mask] = 0.0
    return out


# ---------------------------------------------------------------------------
# LOBO runner
# ---------------------------------------------------------------------------


def run_lobo(
    model_factory: Callable,
    X: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cidx: np.ndarray,
    device,
    *,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 50,
    scheduler: str | None = "cosine",
    progress: bool = True,
) -> list[dict]:
    """Leave-One-Battery-Out cross-validation.

    For each of the 13 batteries:
      - Hold that battery out as the test set.
      - Hold the last controlled battery still in train as the inner-val set.
      - Fit global_scale on inner-train only (no leakage).
      - Train model via sequence.train_model (returns loss curves).
      - Collect per-fold scatter + loss-curve data.

    Returns a list of 13 fold dicts, each with:
        battery, y_te, y_pred, cidx_te, train_curve, val_curve, best_epoch, metrics.
    """
    import torch
    from sequence import train_model  # reuse verified training loop
    from vg_models import predict_vg

    all_bids = sorted(set(groups))
    ctrl_bids = sorted(b for b in all_bids if b in CONTROLLED_BIDS)

    folds: list[dict] = []

    for fold_idx, held_out in enumerate(all_bids):
        is_test  = groups == held_out
        is_avail = ~is_test

        # Inner-val: last controlled battery still in train, else last train battery
        avail_ctrl = [b for b in ctrl_bids if b != held_out]
        val_bid    = avail_ctrl[-1] if avail_ctrl else sorted(set(groups[is_avail]))[-1]
        is_val     = is_avail & (groups == val_bid)
        is_train   = is_avail & ~is_val

        X_tr, mask_tr, y_tr = X[is_train], mask[is_train], y[is_train]
        X_val, mask_val, y_val = X[is_val], mask[is_val], y[is_val]
        X_te, mask_te, y_te = X[is_test], mask[is_test], y[is_test]

        # Scale — fit on inner-train valid rows only; re-apply neutral fill after
        X_tr_s, X_val_s, X_te_s = global_scale(X_tr, X_val, X_te, mask_tr)
        X_tr_s  = _reapply_fill(X_tr_s,  mask_tr)
        X_val_s = _reapply_fill(X_val_s, mask_val)
        X_te_s  = _reapply_fill(X_te_s,  mask_te)

        torch.manual_seed(fold_idx * 31 + 42)
        model = model_factory().to(device)

        model, train_curve, val_curve, best_ep = train_model(
            model,
            X_tr_s,  mask_tr,  y_tr,
            X_val_s, mask_val, y_val,
            device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            scheduler=scheduler,
            progress=progress,
        )

        y_pred = predict_vg(model, X_te_s, mask_te, device)
        metrics = lobo_metrics(y_te, y_pred, cidx[is_test], y_tr.mean())

        print(f"  [{fold_idx+1:2d}/13] held={held_out:6s}  "
              f"MAE={metrics['mae']:.4f}  R²={metrics['r2']:.3f}  "
              f"best_ep={best_ep}")

        folds.append({
            "battery":    held_out,
            "y_te":       y_te,
            "y_pred":     y_pred,
            "cidx_te":    cidx[is_test],
            "train_curve": train_curve,
            "val_curve":   val_curve,
            "best_epoch":  best_ep,
            "metrics":     metrics,
        })

    return folds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def lobo_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cidx: np.ndarray,
    train_mean: float,
) -> dict:
    """Per-fold evaluation metrics in fractional-rated capacity space."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import warnings

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))

    naive_mae = float(np.abs(y_true - train_mean).mean())
    skill     = float(1.0 - mae / naive_mae) if naive_mae > 0 else float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp_corr, _ = spearmanr(y_true, y_pred)

    return {
        "mae":       mae,
        "rmse":      rmse,
        "r2":        r2,
        "skill":     skill,
        "spearman":  float(sp_corr) if not np.isnan(sp_corr) else float("nan"),
        "naive_mae": naive_mae,
        "n":         len(y_true),
    }


def aggregate(folds: list[dict]) -> dict:
    """Aggregate per-fold metrics: mean ± std, NaN-filtered."""
    keys = ["mae", "rmse", "r2", "skill", "spearman"]
    out: dict = {}
    for k in keys:
        vals = [f["metrics"][k] for f in folds if not np.isnan(f["metrics"][k])]
        out[k]          = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    out["n_folds"] = len(folds)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def per_fold_scatter(
    folds: list[dict],
    title: str,
    save_path: Path | None = None,
) -> None:
    """Per-fold scatter: predicted vs actual fractional capacity.

    Controlled batteries plotted in blue, randomized in orange.
    Diagonal y=x reference line.  Per-subplot MAE/R² annotation.
    """
    import matplotlib.pyplot as plt

    n  = len(folds)
    nc = 4
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.5 * nr),
                             constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, fold in zip(axes_flat, folds):
        bid   = fold["battery"]
        color = "steelblue" if bid in CONTROLLED_BIDS else "darkorange"
        ax.scatter(fold["y_te"], fold["y_pred"], s=12, alpha=0.7, color=color)
        lo = min(fold["y_te"].min(), fold["y_pred"].min()) - 0.02
        hi = max(fold["y_te"].max(), fold["y_pred"].max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        m = fold["metrics"]
        ax.set_title(f"{bid}\nMAE={m['mae']:.4f}  R²={m['r2']:.3f}", fontsize=8)
        ax.set_xlabel("True Ct/C₀", fontsize=7)
        ax.set_ylabel("Pred Ct/C₀", fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[len(folds):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=11)
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def per_fold_loss_curves(
    folds: list[dict],
    title: str,
    save_path: Path | None = None,
) -> None:
    """Per-fold train/val L1 loss curves with best-epoch marker.

    Allows visual diagnosis of overfitting per held-out battery.
    """
    import matplotlib.pyplot as plt

    n  = len(folds)
    nc = 4
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3 * nr),
                             constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, fold in zip(axes_flat, folds):
        tr = fold["train_curve"]
        vl = fold["val_curve"]
        ep = fold["best_epoch"]
        epochs = range(1, len(tr) + 1)
        ax.plot(epochs, tr, label="train", lw=1.2)
        ax.plot(epochs, vl, label="val",   lw=1.2)
        if ep > 0:
            ax.axvline(ep, color="gray", lw=0.8, ls="--")
        ax.set_title(f"{fold['battery']}  (best ep={ep})", fontsize=8)
        ax.set_xlabel("Epoch", fontsize=7)
        ax.set_ylabel("L1 loss", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)

    for ax in axes_flat[len(folds):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=11)
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
