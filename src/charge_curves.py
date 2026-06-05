"""Within-cycle charge-curve extraction utilities for notebooks 05–06.

Extracts the CV-phase charge segment preceding each retained discharge cycle,
resamples it to a fixed-length 3-channel (V, I, T) tensor, and joins to
per-cycle SOH labels from the processed CSVs.

Reuses load_controlled / load_randomized from src/battery.py (raw .mat loaders)
and mirrors the same discharge-selection and cycle_index re-indexing logic so
that (battery_id, cycle_index) keys match the processed CSVs exactly.

Extraction window
-----------------
We capture the CV phase only: from the first point where V ≥ v_cv (~4.15 V, the
CC→CV transition) to the last point where |I| ≥ i_cut (end of the CV taper).
This supersedes the old v_lo=3.8 V CC-onset start gate.

Why CV-only?
------------
RW1 and RW9 start their reference charges from a random state-of-charge (SOC),
so v0 can be anywhere from 3.5 V to 4.23 V.  With the old CC-onset gate, the
same battery at the same SOH produced a different curve shape depending on where
the charge happened to start — SOC noise uncorrelated with the label.  Under
LOBO, the model trained on mixed SOCs cannot generalise to a held-out battery.

The CV phase (constant voltage ~4.2 V, current decaying 2 A → 0) is present in
100% of reference charges regardless of starting SOC, and it is the primary SOH
signal: a degraded cell reaches the CV cutoff faster, bending the I-decay curve.
Anchoring there makes the representation SOC-invariant across both cohorts.

Leakage note
------------
Input = charge-side V/I/T.  Uniform-time resampling over the CV segment discards
absolute CV duration, keeping only the current-decay shape.  That shape is the
SOH signal; absolute duration would be a mild label proxy.  Label = discharge-
derived capacity_retention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from battery import load_controlled, load_randomized


# ---------------------------------------------------------------------------
# CC+CV slice and resample
# ---------------------------------------------------------------------------


def slice_resample(
    V: np.ndarray,
    I: np.ndarray,
    T: np.ndarray,
    v_cv: float,
    i_cut: float,
    n_points: int,
    i_min: float = 1.0,
    v_phys_max: float = 4.3,
    min_pts: int = 10,
    resample_axis: str = "time",
) -> np.ndarray | None:
    """Extract the CV-phase charge segment and resample to n_points.

    Segment: from the first index where V ≥ v_cv (CC→CV transition, ~4.15 V)
    to the last index where |I| ≥ i_cut (end of CV taper).  The CC ramp is
    intentionally excluded so the representation is invariant to starting SOC.

    Parameters
    ----------
    V, I, T : 1-D arrays
        Voltage (V), current (A, negative convention OK), temperature (°C)
        from a single charge step.
    v_cv : float
        CV-onset voltage threshold.  The first sample at or above this voltage
        is taken as the start of the CV phase (default 4.15 V).
    i_cut : float
        Current cutoff in A; the last sample with |I| ≥ i_cut defines the
        end of the CV taper.  Typical value: 0.04 A (≈ C/50 for a 2 Ah cell).
    n_points : int
        Resample length (fixed for all curves).
    i_min : float
        Minimum peak |I| required in the sliced segment.  Steps whose current
        never exceeds this (e.g. near-empty top-up stubs) are rejected.
        Default 1.0 A.
    v_phys_max : float
        Physical upper voltage bound for Li-ion.  Steps with any sample above
        this value are rejected as sensor-corrupted.  Default 4.3 V.
    min_pts : int
        Minimum number of raw samples required in the sliced segment after
        applying the CV onset and i_cut gates.  Default 10.
    resample_axis : {"time", "current"}
        "time"    — uniform resampling over sample index (default).  The
                    current-decay rate (how fast |I| falls) is the primary SOH
                    signal encoded in the I channel.
        "current" — resample V and T at n_points evenly spaced |I| levels from
                    I_peak down to i_cut.  The I channel becomes a fixed
                    monotone grid (no information); V and T at each current
                    level encode the V–I characteristic and thermal state, which
                    carry complementary SOH signal and improve resolution at
                    high SOH where time-domain curves look nearly identical.

    Returns
    -------
    np.ndarray of shape (n_points, 3) [V, |I|, T], float32, or None if any
    quality gate fails.
    """
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    T = np.asarray(T, dtype=float)

    # Physical guard: reject sensor-corrupted steps (e.g. 8.39 V spike in B0005).
    if np.nanmax(V) > v_phys_max:
        return None

    # CV onset: find the first index where V >= v_cv AND the next sample also >= v_cv.
    # The two-consecutive-samples requirement skips single-sample DCIR-pulse artifacts
    # that appear at the start of charge steps in the NASA dataset (brief current spike
    # + V dip to ~3.78 V before CC/CV charging begins).
    above_cv = np.where(V >= v_cv)[0]
    if len(above_cv) == 0:
        return None
    i_start = None
    for k in above_cv:
        if k + 1 < len(V) and V[k + 1] >= v_cv:
            i_start = int(k)
            break
    if i_start is None:
        return None

    # End: last index from i_start where |I| is still above the cutoff.
    abs_I_seg = np.abs(I[i_start:])
    above_cut = np.where(abs_I_seg >= i_cut)[0]
    if len(above_cut) == 0:
        return None
    i_end = i_start + int(above_cut[-1])

    if i_end <= i_start:
        return None

    seg_V = V[i_start : i_end + 1]
    seg_I = I[i_start : i_end + 1]
    seg_T = T[i_start : i_end + 1]

    # Stub guard: reject near-empty top-up charges (no real SOH signal).
    if np.abs(seg_I).max() < i_min:
        return None
    if len(seg_V) < min_pts:
        return None

    abs_I = np.abs(seg_I)

    if resample_axis == "current":
        # Resample V and T at n_points evenly spaced current levels from I_peak → i_cut.
        # |I| during CV is approximately monotone decreasing; enforce it with a running
        # minimum so np.interp gets a strictly valid (non-decreasing, reversed) x-axis.
        abs_I_mono = np.minimum.accumulate(abs_I)   # force monotone non-increasing
        I_peak = float(abs_I_mono[0])
        I_grid = np.linspace(I_peak, i_cut, n_points)  # decreasing: high → low current

        # np.interp requires increasing x: flip both axes, query, then flip result back.
        x_inc   = abs_I_mono[::-1]       # now non-decreasing (low → high)
        V_inc   = seg_V[::-1]
        T_inc   = seg_T[::-1]
        I_q_inc = I_grid[::-1]           # query points, increasing

        r_V = np.interp(I_q_inc, x_inc, V_inc)[::-1]
        r_T = np.interp(I_q_inc, x_inc, T_inc)[::-1]
        r_I = I_grid                     # fixed grid — model learns to ignore this channel
    else:
        # Uniform resampling over sample index (time).
        old_idx = np.linspace(0, len(seg_V) - 1, len(seg_V))
        new_idx = np.linspace(0, len(seg_V) - 1, n_points)
        r_V = np.interp(new_idx, old_idx, seg_V)
        # |I|: controlled charges with positive current, RW with negative; abs normalises.
        r_I = np.interp(new_idx, old_idx, abs_I)
        r_T = np.interp(new_idx, old_idx, seg_T)

    return np.stack([r_V, r_I, r_T], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Controlled battery extraction
# ---------------------------------------------------------------------------


def extract_controlled(
    cycles: list[dict],
    battery_id: str,
    v_cv: float,
    i_cut: float,
    n_points: int,
    i_min: float = 1.0,
    v_phys_max: float = 4.3,
    min_pts: int = 10,
    resample_axis: str = "time",
) -> list[dict]:
    """Extract charge-curve tensors for one controlled battery.

    Mirrors cycles_to_df from battery.py: only processes discharge cycles,
    resets cycle_index to range(len(retained_discharges)) after the same
    capacity > 0.5*baseline filter — so the emitted cycle_index matches the
    processed CSV exactly.

    Each returned dict has keys:
        battery_id, cycle_index, tensor (n_points, 3) or None (window not covered).
    """
    # --- pass 1: collect (discharge_cycle, preceding_charge_cycle) pairs ---
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

    # --- capacity filter (mirrors cycles_to_df lines 85-89) ---
    cap_vals = []
    for dis, _ in pairs:
        d = dis["data"]
        time = np.asarray(d["Time"], dtype=float)
        current = np.asarray(d["Current_measured"], dtype=float)
        cap = float(np.trapezoid(np.abs(current), time) / 3600)
        cap_vals.append(cap)

    n_warmup = max(5, int(0.10 * len(pairs)))
    baseline = max(cap_vals[:n_warmup])
    kept = [(dis, chg) for (dis, chg), cap in zip(pairs, cap_vals) if cap > 0.5 * baseline]

    if not kept:
        return []

    # --- pass 2: extract tensors, assign re-indexed cycle_index ---
    records = []
    for new_idx, (dis, chg) in enumerate(kept):
        tensor = None
        if chg is not None:
            d = chg["data"]
            V = np.asarray(d["Voltage_measured"], dtype=float)
            I = np.asarray(d["Current_measured"], dtype=float)
            Temp = np.asarray(d["Temperature_measured"], dtype=float)
            tensor = slice_resample(V, I, Temp, v_cv, i_cut, n_points,
                                    i_min=i_min, v_phys_max=v_phys_max, min_pts=min_pts,
                                    resample_axis=resample_axis)

        records.append({
            "battery_id": battery_id,
            "cycle_index": new_idx,
            "tensor": tensor,
        })

    return records


# ---------------------------------------------------------------------------
# Randomized battery extraction
# ---------------------------------------------------------------------------


def extract_randomized(
    steps: list[dict],
    battery_id: str,
    v_cv: float,
    i_cut: float,
    n_points: int,
    i_min: float = 1.0,
    v_phys_max: float = 4.3,
    min_pts: int = 10,
    resample_axis: str = "time",
) -> list[dict]:
    """Extract charge-curve tensors for one randomized battery.

    Mirrors rw_steps_to_df from battery.py: keeps only reference-discharge
    steps that pass quality gates, applies iloc[::2] decimation, and resets
    cycle_index = range(n) — matching the processed CSV exactly.

    Each reference-discharge is paired with the preceding reference-charge
    step (type='C' + 'reference' in comment).

    Each returned dict has keys:
        battery_id, cycle_index, tensor (n_points, 3) or None.
    """
    # --- collect raw reference (discharge, charge) pairs with quality gates ---
    raw_pairs: list[tuple[dict, dict | None]] = []
    last_ref_charge: dict | None = None

    for step in steps:
        stype = str(step.get("type", "")).upper()
        comment = str(step.get("comment", "")).lower()
        is_ref = "reference" in comment

        if stype == "C" and is_ref:
            last_ref_charge = step

        elif stype == "D" and is_ref:
            # Quality gates (mirroring rw_steps_to_df lines 129-137).
            time = np.asarray(step["relativeTime"], dtype=float)
            current = np.asarray(step["current"], dtype=float)

            if len(time) < 2:
                continue
            if np.abs(current).mean() < 0.5:
                continue
            if float(time[-1] - time[0]) < 1200:
                continue

            raw_pairs.append((step, last_ref_charge))

    if not raw_pairs:
        return []

    # --- iloc[::2] decimation (mirrors rw_steps_to_df line 170) ---
    decimated = raw_pairs[::2]

    # --- extract tensors, assign re-indexed cycle_index ---
    records = []
    for new_idx, (dis, chg) in enumerate(decimated):
        tensor = None
        if chg is not None:
            V = np.asarray(chg["voltage"], dtype=float)
            I = np.asarray(chg["current"], dtype=float)
            Temp = np.asarray(chg["temperature"], dtype=float)
            tensor = slice_resample(V, I, Temp, v_cv, i_cut, n_points,
                                    i_min=i_min, v_phys_max=v_phys_max, min_pts=min_pts,
                                    resample_axis=resample_axis)

        records.append({
            "battery_id": battery_id,
            "cycle_index": new_idx,
            "tensor": tensor,
        })

    return records


# ---------------------------------------------------------------------------
# Build full tensor dataset (all 13 batteries)
# ---------------------------------------------------------------------------

# Paths to the retained batteries' .mat files, relative to the project root.
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


# ---------------------------------------------------------------------------
# CC+CV slice with time channel (nb08)
# ---------------------------------------------------------------------------


def slice_resample_windowed(
    V: np.ndarray,
    I: np.ndarray,
    T: np.ndarray,
    Time: np.ndarray,
    v_cv: float,
    i_cut: float,
    n_points: int,
    v_start: float | None = None,
    i_min: float = 1.0,
    v_phys_max: float = 4.3,
    min_pts: int = 10,
) -> tuple[np.ndarray | None, dict]:
    """Extract CC+CV (or CV-only) charge segment with a 4-channel tensor.

    Channels: [V, |I|, T, t_elapsed_s]
    Channel 4 = real elapsed seconds from segment start (restores duration).

    Parameters
    ----------
    v_start : float | None
        Anchor voltage for CC-ramp start.  None → CV-only (start at v_cv,
        D1 mode).  Float → start at first upward crossing of v_start (D2
        mode); falls back to v_cv if the charge began above the anchor.

    Returns
    -------
    (tensor (n_points, 4) float32 or None, cc_scalars dict)
    cc_scalars keys: 'cc_dt_anchor_to_cv' (s), 'cc_slope' (V/s).
    Both are 0.0 when no CC ramp is captured (v_start=None or fallback).
    """
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    T = np.asarray(T, dtype=float)
    Time = np.asarray(Time, dtype=float)
    _empty = {"cc_dt_anchor_to_cv": 0.0, "cc_slope": 0.0}

    if np.nanmax(V) > v_phys_max:
        return None, _empty

    # CV onset (two-consecutive-sample rule, same as slice_resample)
    above_cv = np.where(V >= v_cv)[0]
    if len(above_cv) == 0:
        return None, _empty
    i_cv = None
    for k in above_cv:
        if k + 1 < len(V) and V[k + 1] >= v_cv:
            i_cv = int(k)
            break
    if i_cv is None:
        return None, _empty

    cc_dt = 0.0
    cc_slope = 0.0

    if v_start is None:
        # D1 mode: CV-only with time channel
        i_start = i_cv
    elif V[0] >= v_start:
        # Charge began above anchor — no CC ramp to capture
        i_start = i_cv
    else:
        # D2 mode: find first upward crossing of v_start (two-consecutive rule)
        above_vs = np.where(V >= v_start)[0]
        i_anchor = None
        for k in above_vs:
            if k + 1 < len(V) and V[k + 1] >= v_start:
                i_anchor = int(k)
                break
        if i_anchor is None or i_anchor >= i_cv:
            i_start = i_cv  # fall back to CV onset
        else:
            i_start = i_anchor
            dt = float(Time[i_cv] - Time[i_anchor])
            cc_dt = max(0.0, dt)
            if cc_dt > 0.0:
                cc_slope = float((V[i_cv] - V[i_anchor]) / cc_dt)

    # End: last sample from i_start where |I| >= i_cut (unchanged logic)
    abs_I_seg = np.abs(I[i_start:])
    above_cut = np.where(abs_I_seg >= i_cut)[0]
    if len(above_cut) == 0:
        return None, {"cc_dt_anchor_to_cv": cc_dt, "cc_slope": cc_slope}
    i_end = i_start + int(above_cut[-1])
    if i_end <= i_start:
        return None, {"cc_dt_anchor_to_cv": cc_dt, "cc_slope": cc_slope}

    seg_V = V[i_start: i_end + 1]
    seg_I = I[i_start: i_end + 1]
    seg_T = T[i_start: i_end + 1]
    seg_time = Time[i_start: i_end + 1]

    if np.abs(seg_I).max() < i_min or len(seg_V) < min_pts:
        return None, {"cc_dt_anchor_to_cv": cc_dt, "cc_slope": cc_slope}

    abs_I = np.abs(seg_I)
    t_elapsed = seg_time - seg_time[0]  # seconds from segment start

    old_idx = np.linspace(0, len(seg_V) - 1, len(seg_V))
    new_idx = np.linspace(0, len(seg_V) - 1, n_points)
    r_V  = np.interp(new_idx, old_idx, seg_V)
    r_I  = np.interp(new_idx, old_idx, abs_I)
    r_T  = np.interp(new_idx, old_idx, seg_T)
    r_te = np.interp(new_idx, old_idx, t_elapsed)

    tensor = np.stack([r_V, r_I, r_T, r_te], axis=-1).astype(np.float32)
    return tensor, {"cc_dt_anchor_to_cv": cc_dt, "cc_slope": cc_slope}


def extract_controlled_cc(
    cycles: list[dict],
    battery_id: str,
    v_cv: float,
    i_cut: float,
    n_points: int,
    v_start: float | None = None,
    i_min: float = 1.0,
    v_phys_max: float = 4.3,
    min_pts: int = 10,
) -> list[dict]:
    """Like extract_controlled but calls slice_resample_windowed (4-channel tensor).

    Returned dicts: battery_id, cycle_index, tensor (n_points,4)|None,
    cc_dt_anchor_to_cv, cc_slope.
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

    cap_vals = []
    for dis, _ in pairs:
        d = dis["data"]
        time = np.asarray(d["Time"], dtype=float)
        current = np.asarray(d["Current_measured"], dtype=float)
        cap_vals.append(float(np.trapezoid(np.abs(current), time) / 3600))

    n_warmup = max(5, int(0.10 * len(pairs)))
    baseline = max(cap_vals[:n_warmup])
    kept = [(dis, chg) for (dis, chg), cap in zip(pairs, cap_vals) if cap > 0.5 * baseline]

    if not kept:
        return []

    records = []
    for new_idx, (dis, chg) in enumerate(kept):
        tensor, cc_sc = None, {"cc_dt_anchor_to_cv": 0.0, "cc_slope": 0.0}
        if chg is not None:
            d = chg["data"]
            V    = np.asarray(d["Voltage_measured"], dtype=float)
            I    = np.asarray(d["Current_measured"], dtype=float)
            Temp = np.asarray(d["Temperature_measured"], dtype=float)
            TT   = np.asarray(d["Time"], dtype=float)
            tensor, cc_sc = slice_resample_windowed(
                V, I, Temp, TT, v_cv, i_cut, n_points,
                v_start=v_start, i_min=i_min, v_phys_max=v_phys_max, min_pts=min_pts,
            )
        records.append({
            "battery_id": battery_id,
            "cycle_index": new_idx,
            "tensor": tensor,
            **cc_sc,
        })
    return records


def extract_randomized_cc(
    steps: list[dict],
    battery_id: str,
    v_cv: float,
    i_cut: float,
    n_points: int,
    v_start: float | None = None,
    i_min: float = 1.0,
    v_phys_max: float = 4.3,
    min_pts: int = 10,
) -> list[dict]:
    """Like extract_randomized but calls slice_resample_windowed (4-channel tensor)."""
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
            current = np.asarray(step["current"], dtype=float)
            if len(time) < 2 or np.abs(current).mean() < 0.5:
                continue
            if float(time[-1] - time[0]) < 1200:
                continue
            raw_pairs.append((step, last_ref_charge))

    if not raw_pairs:
        return []

    decimated = raw_pairs[::2]
    records = []
    for new_idx, (dis, chg) in enumerate(decimated):
        tensor, cc_sc = None, {"cc_dt_anchor_to_cv": 0.0, "cc_slope": 0.0}
        if chg is not None:
            V    = np.asarray(chg["voltage"], dtype=float)
            I    = np.asarray(chg["current"], dtype=float)
            Temp = np.asarray(chg["temperature"], dtype=float)
            TT   = np.asarray(chg["relativeTime"], dtype=float)
            tensor, cc_sc = slice_resample_windowed(
                V, I, Temp, TT, v_cv, i_cut, n_points,
                v_start=v_start, i_min=i_min, v_phys_max=v_phys_max, min_pts=min_pts,
            )
        records.append({
            "battery_id": battery_id,
            "cycle_index": new_idx,
            "tensor": tensor,
            **cc_sc,
        })
    return records


def build_dataset_cc(
    project_root: Path,
    v_cv: float,
    i_cut: float,
    n_points: int,
    v_start: float | None = None,
    i_min: float = 1.0,
    v_phys_max: float = 4.3,
    min_pts: int = 10,
) -> pd.DataFrame:
    """Extract 4-channel charge-curve tensors for all 13 retained batteries.

    v_start=None → CV-only + time (D1).
    v_start=float → anchored CC+CV + time (D2 / headline).

    Returns DataFrame: battery_id, cycle_index, tensor (or None),
    cc_dt_anchor_to_cv, cc_slope.
    """
    all_records: list[dict] = []

    for bid, rel_path in CONTROLLED_MATS.items():
        path = project_root / rel_path
        cycles = load_controlled(path)
        records = extract_controlled_cc(cycles, bid, v_cv, i_cut, n_points,
                                        v_start=v_start, i_min=i_min,
                                        v_phys_max=v_phys_max, min_pts=min_pts)
        all_records.extend(records)
        print(f"  {bid}: {len(records)} cycles")

    for bid, rel_path in RANDOMIZED_MATS.items():
        path = project_root / rel_path
        steps = load_randomized(path)
        records = extract_randomized_cc(steps, bid, v_cv, i_cut, n_points,
                                        v_start=v_start, i_min=i_min,
                                        v_phys_max=v_phys_max, min_pts=min_pts)
        all_records.extend(records)
        print(f"  {bid}: {len(records)} cycles")

    return pd.DataFrame(all_records)


def build_dataset(
    project_root: Path,
    v_cv: float,
    i_cut: float,
    n_points: int,
    i_min: float = 1.0,
    v_phys_max: float = 4.3,
    min_pts: int = 10,
    resample_axis: str = "time",
) -> pd.DataFrame:
    """Extract charge-curve tensors for all 13 retained batteries.

    Parameters
    ----------
    project_root : Path
        Repo root (parent of data/, src/).
    v_cv : float
        CV-onset voltage threshold; start of extracted segment (default 4.15 V).
    i_cut : float
        Current cutoff in A; end of CV taper (default 0.04 A).
    n_points : int
        Resample length per curve.
    i_min : float
        Minimum peak |I| in segment; rejects near-empty stubs (default 1.0 A).
    v_phys_max : float
        Physical voltage ceiling; rejects sensor-corrupted cycles (default 4.3 V).
    min_pts : int
        Minimum raw samples in CV segment (default 10).
    resample_axis : {"time", "current"}
        Resampling axis — see slice_resample for full description.

    Returns
    -------
    DataFrame with columns:
        battery_id, cycle_index, tensor (ndarray or None)
    One row per retained discharge cycle (same indexing as the processed CSVs).
    """
    all_records: list[dict] = []

    for bid, rel_path in CONTROLLED_MATS.items():
        path = project_root / rel_path
        cycles = load_controlled(path)
        records = extract_controlled(cycles, bid, v_cv, i_cut, n_points,
                                     i_min=i_min, v_phys_max=v_phys_max, min_pts=min_pts,
                                     resample_axis=resample_axis)
        all_records.extend(records)
        print(f"  {bid}: {len(records)} cycles extracted")

    for bid, rel_path in RANDOMIZED_MATS.items():
        path = project_root / rel_path
        steps = load_randomized(path)
        records = extract_randomized(steps, bid, v_cv, i_cut, n_points,
                                     i_min=i_min, v_phys_max=v_phys_max, min_pts=min_pts,
                                     resample_axis=resample_axis)
        all_records.extend(records)
        print(f"  {bid}: {len(records)} cycles extracted")

    return pd.DataFrame(all_records)
