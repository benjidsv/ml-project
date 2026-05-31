"""Shared data-loading and feature-engineering utilities for the battery health project."""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io


def _datenum(tv) -> float:
    """Convert NASA MATLAB date vector [y, mo, d, h, mi, s] to POSIX timestamp."""
    y, mo, d, h, mi, s = tv
    base = datetime.datetime(int(y), int(mo), int(d))
    return (
        base + datetime.timedelta(hours=float(h), minutes=float(mi), seconds=float(s))
    ).timestamp()


def discharge_energy_wh(voltage, current, time) -> float:
    """Discharge energy via V·|I| integration (Wh).

    Captures both capacity loss *and* voltage sag due to rising internal resistance,
    so it declines faster than capacity alone — a richer efficiency proxy for grid storage.
    """
    V = np.asarray(voltage, dtype=float)
    I = np.asarray(current, dtype=float)
    T = np.asarray(time, dtype=float)
    return float(np.trapezoid(V * np.abs(I), T) / 3600)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_controlled(path: Path) -> list[dict]:
    mat = scipy.io.loadmat(path, simplify_cells=True)
    key = next(k for k in mat if not k.startswith("_"))
    return mat[key]["cycle"]


def cycles_to_df(
    cycles: list[dict], battery_id: str, cutoff_v: float, temp_condition: str
) -> pd.DataFrame:
    """One row per discharge cycle. Capacity computed via current integration."""
    rows = []
    discharge_idx = 0

    for cycle in cycles:
        if cycle["type"] != "discharge":
            continue
        d = cycle["data"]
        time = np.asarray(d["Time"], dtype=float)
        current = np.asarray(d["Current_measured"], dtype=float)
        voltage = np.asarray(d["Voltage_measured"], dtype=float)
        temp = np.asarray(d["Temperature_measured"], dtype=float)

        rows.append({
            "battery_id": battery_id,
            "cutoff_v": cutoff_v,
            "temp_condition": temp_condition,
            "cycle_index": discharge_idx,
            "timestamp": _datenum(cycle["time"]),
            "ambient_temp": float(cycle["ambient_temperature"]),
            "capacity_ahr": float(np.trapezoid(np.abs(current), time) / 3600),
            "energy_wh": discharge_energy_wh(voltage, current, time),
            "r_proxy": dcir_proxy_controlled(cycle["data"]),
            "voltage_min": float(voltage.min()),
            "voltage_mean": float(voltage.mean()),
            "temp_mean": float(temp.mean()),
            "temp_max": float(temp.max()),
            "duration_s": float(time[-1] - time[0]),
        })
        discharge_idx += 1

    df = pd.DataFrame(rows)

    df["avg_current_a"] = df["capacity_ahr"] * 3600 / df["duration_s"]

    # B0042–B0044 interleave 1 A and 4 A discharge cycles; keep only 4 A primary protocol.
    if temp_condition == "cold":
        df = df[df["avg_current_a"] >= 1.8].reset_index(drop=True)

    n_warmup = max(5, int(0.10 * len(df)))
    baseline = df["capacity_ahr"].iloc[:n_warmup].max()

    df = df[df["capacity_ahr"] > 0.5 * baseline].reset_index(drop=True)
    df["cycle_index"] = range(len(df))

    n = len(df)
    df["rul"] = (n - 1 - df["cycle_index"]) / (
        n - 1
    )  # legacy (dataset-relative); replaced in prep
    df["capacity_retention"] = df["capacity_ahr"] / baseline
    df["_baseline"] = baseline

    # Energy retention: same baseline convention (max over first 10 % of cycles).
    energy_baseline = df["energy_wh"].iloc[:n_warmup].max()
    df["energy_retention"] = df["energy_wh"] / energy_baseline
    df["_energy_baseline"] = energy_baseline
    return df


def load_randomized(path: Path) -> list[dict]:
    mat = scipy.io.loadmat(path, simplify_cells=True)
    key = next(k for k in mat if not k.startswith("_"))
    return mat[key]["step"]


def rw_steps_to_df(
    steps: list[dict], battery_id: str, rw_condition: str
) -> pd.DataFrame:
    """One row per reference discharge step."""
    rows = []
    ref_idx = 0
    prev_step = None

    for i, step in enumerate(steps):
        comment = str(step.get("comment", "")).lower()
        stype = str(step.get("type", "")).upper()

        if stype == "D" and "reference" in comment:
            time = np.asarray(step["relativeTime"], dtype=float)
            current = np.asarray(step["current"], dtype=float)
            voltage = np.asarray(step["voltage"], dtype=float)
            temp = np.asarray(step["temperature"], dtype=float)

            if len(time) < 2:
                prev_step = step
                continue
            if np.abs(current).mean() < 0.5:
                prev_step = step
                continue
            if float(time[-1] - time[0]) < 1200:
                prev_step = step
                continue

            # Temperature: clip to valid physical range; RW2 raw temps are valid — widen clip slightly.
            temp_valid = temp[(temp > -20) & (temp < 80)]
            temp_mean = (
                float(temp_valid.mean()) if len(temp_valid) > 0 else float("nan")
            )
            temp_max = float(temp_valid.max()) if len(temp_valid) > 0 else float("nan")

            r_proxy = dcir_proxy_rw(step, prev_step)

            rows.append({
                "battery_id": battery_id,
                "rw_condition": rw_condition,
                "cycle_index": ref_idx,
                "capacity_ahr": float(np.trapezoid(np.abs(current), time) / 3600),
                "energy_wh": discharge_energy_wh(voltage, current, time),
                "voltage_min": float(voltage.min()),
                "voltage_mean": float(voltage.mean()),
                "temp_mean": temp_mean,
                "temp_max": temp_max,
                "duration_s": float(time[-1] - time[0]),
                "r_proxy": r_proxy,
            })
            ref_idx += 1

        prev_step = step

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Drop odd-indexed rows (partial-recovery discharges that produce sawtooth).
    df = df.iloc[::2].reset_index(drop=True)

    n = len(df)
    df["cycle_index"] = range(n)
    df["rul"] = (n - 1 - df["cycle_index"]) / (n - 1)  # legacy
    baseline = df["capacity_ahr"].iloc[0]
    df["capacity_retention"] = df["capacity_ahr"] / baseline
    df["_baseline"] = baseline

    # Energy retention: baseline = first reference cycle (same convention as capacity).
    energy_baseline = df["energy_wh"].iloc[0]
    df["energy_retention"] = df["energy_wh"] / energy_baseline
    df["_energy_baseline"] = energy_baseline
    return df


# ---------------------------------------------------------------------------
# Impedance
# ---------------------------------------------------------------------------


def load_impedance(path: Path) -> pd.DataFrame:
    """One row per impedance cycle: timestamp (posix), Re, Rct."""
    mat = scipy.io.loadmat(path, simplify_cells=True)
    key = next(k for k in mat if not k.startswith("_"))
    cycles = mat[key]["cycle"]

    rows = []
    for c in cycles:
        if c["type"] != "impedance":
            continue
        rows.append({
            "timestamp": _datenum(c["time"]),
            "Re": float(c["data"]["Re"]),
            "Rct": float(c["data"]["Rct"]),
        })

    return pd.DataFrame(rows)


def attach_impedance(
    discharge_df: pd.DataFrame, imp_df: pd.DataFrame, tolerance_h: float = 48.0
) -> pd.DataFrame:
    """Nearest-neighbour join of EIS to discharge cycles by timestamp.

    Cycles more than `tolerance_h` hours from any impedance measurement get NaN;
    remaining NaNs (early cycles) are back-filled from the first available EIS reading.
    """
    imp_times = imp_df["timestamp"].to_numpy()
    re_vals = imp_df["Re"].to_numpy()
    rct_vals = imp_df["Rct"].to_numpy()

    re_out = []
    rct_out = []
    for t in discharge_df["timestamp"]:
        j = int(np.argmin(np.abs(imp_times - t)))
        if abs(imp_times[j] - t) / 3600 > tolerance_h:
            re_out.append(float("nan"))
            rct_out.append(float("nan"))
        else:
            re_out.append(re_vals[j])
            rct_out.append(rct_vals[j])

    df = discharge_df.copy()
    df["Re"] = re_out
    df["Rct"] = rct_out
    # Back-fill NaNs at the start (no impedance measurement yet in early cycles).
    df[["Re", "Rct"]] = df[["Re", "Rct"]].bfill()
    return df


# ---------------------------------------------------------------------------
# Resistance proxy (DCIR)
# ---------------------------------------------------------------------------


def dcir_proxy_controlled(d: dict) -> float:
    """Onset ΔV/ΔI proxy for controlled batteries.

    V_rest = max of the first 3 samples (at rest / load onset).
    V_loaded = median of first 20 loaded samples (|I| > 0.5 A).
    R = (V_rest - V_loaded) / |I_load_median|
    """
    V = np.asarray(d["Voltage_measured"], dtype=float)
    I = np.asarray(d["Current_measured"], dtype=float)
    load = np.abs(I) > 0.5
    if not load.any():
        return float("nan")
    i_load = np.median(np.abs(I[load]))
    v_rest = V[:3].max()
    v_load = np.median(V[load][:20])
    return float((v_rest - v_load) / i_load)


def dcir_proxy_rw(step: dict, prev_step) -> float:
    """Onset ΔV/ΔI proxy for randomized batteries.

    V_rest = last voltage sample of the preceding step (near rest after charge).
    V_loaded = median of first 20 loaded samples of the reference discharge.
    R = (V_rest - V_loaded) / |I_load_median|
    """
    V = np.asarray(step["voltage"], dtype=float)
    I = np.asarray(step["current"], dtype=float)
    load = np.abs(I) > 0.5
    if not load.any():
        return float("nan")
    i_load = np.median(np.abs(I[load]))
    if prev_step is not None:
        v_prev = np.asarray(prev_step["voltage"], dtype=float)
        v_rest = float(v_prev[-1]) if v_prev.size > 0 else V[:3].max()
    else:
        v_rest = V[:3].max()
    v_load = np.median(V[load][:20])
    return float((v_rest - v_load) / i_load)


# ---------------------------------------------------------------------------
# RUL (physically grounded)
# ---------------------------------------------------------------------------


def compute_rul(df: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    """Replace the legacy (dataset-relative) RUL with one grounded in a physical EOL.

    EOL = first cycle where capacity_retention < threshold.
    rul_cycles = eol_index - cycle_index  (clipped to 0 after EOL).
    rul_frac   = rul_cycles / eol_index   (normalised 0-1).

    Batteries that never reach EOL within the dataset are marked rul_censored=True;
    their rul_cycles counts cycles remaining to the dataset end (a lower bound).
    """
    df = df.copy()
    below = df["capacity_retention"] < threshold
    if below.any():
        eol_idx = int(df.loc[below, "cycle_index"].iloc[0])
        censored = False
    else:
        eol_idx = int(df["cycle_index"].iloc[-1])
        censored = True

    df["rul_cycles"] = (eol_idx - df["cycle_index"]).clip(lower=0).astype(int)
    df["rul_frac"] = df["rul_cycles"] / eol_idx if eol_idx > 0 else 0.0
    df["rul_censored"] = censored
    df["eol_index"] = eol_idx
    # Drop the old dataset-relative rul column to avoid confusion.
    df.drop(columns=["rul"], inplace=True, errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(
    df: pd.DataFrame, windows: list = [5, 10, 15, 20]
) -> pd.DataFrame:
    """Add causal rolling features (no future leakage) per battery series."""
    df = df.copy().sort_values("cycle_index").reset_index(drop=True)

    for col in ["voltage_mean", "temp_mean", "r_proxy"]:
        if col not in df.columns:
            continue
        for window in windows:
            df[f"{col}_{window}_roll_mean"] = (
                df[col].rolling(window, min_periods=1).mean()
            )
            df[f"{col}_{window}_roll_std"] = (
                df[col].rolling(window, min_periods=1).std().fillna(0)
            )

    # Capacity fade rate (Ahr lost per cycle, smoothed).
    for window in windows:
        df[f"cap_fade_rate_{window}"] = (
            df["capacity_ahr"].diff().rolling(window, min_periods=1).mean().fillna(0)
        )

    if "r_proxy" in df.columns:
        for window in windows:
            df[f"r_proxy_rate_{window}"] = (
                df["r_proxy"].diff().rolling(window, min_periods=1).mean().fillna(0)
            )

    # Normalised cycle position (0 = start, 1 = last observed cycle).
    max_cyc = df["cycle_index"].max()
    df["cycle_norm"] = df["cycle_index"] / max_cyc if max_cyc > 0 else 0.0

    # Capacity delta from battery-specific baseline.
    baseline = df["_baseline"].iloc[0]
    df["cap_delta_from_baseline"] = df["capacity_ahr"] - baseline

    return df
