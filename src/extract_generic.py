"""Generic timeseries adapters for CALCE (Arbin .xls) and BatteryArchive CSV cells.

These loaders feed into vg_extended.build_extended_dataset for non-NASA cells.
NASA cells continue to use battery.load_controlled / load_randomized (unchanged).

All paths through this module ultimately call voltage_grid.resample_voltage_grid
without modifying it, then post-normalise the |I| channel to C-rate.

Column conventions after normalisation (internal standard):
    cycle_index     int       — globally unique cycle number within the cell
    time_s          float     — test time in seconds (monotone increasing)
    current_a       float     — current in Amps (positive = charge)
    voltage_v       float     — terminal voltage in V
    temperature_c   float     — cell temperature in °C (or imputed ambient)
    discharge_cap_ah float    — discharge capacity accumulated this cycle (Ah)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import fastexcel
from pathlib import Path

from src.voltage_grid import resample_voltage_grid, V_LO, V_HI, N_GRID


# ---------------------------------------------------------------------------
# Column maps — raw source names → internal standard name
# ---------------------------------------------------------------------------

# Standard Arbin BT2000/BT2143 .xls output (CALCE CS2 / CX2)
CALCE_ARBIN_COLMAP: dict[str, str] = {
    "cycle_index": "Cycle_Index",
    "time_s": "Test_Time(s)",
    "current_a": "Current(A)",
    "voltage_v": "Voltage(V)",
    "discharge_cap_ah": "Discharge_Capacity(Ah)",
    # No temperature column in CALCE — synthesised from ambient_c.
}

# Standard BatteryArchive *_timeseries.csv
BATTERYARCHIVE_COLMAP: dict[str, str] = {
    "cycle_index": "Cycle_Index",
    "time_s": "Test_Time(s)",
    "current_a": "Current(A)",
    "voltage_v": "Voltage(V)",
    "temperature_c": "Cell_Temperature(C)",
    "discharge_cap_ah": "Discharge_Capacity(Ah)",
}


# ---------------------------------------------------------------------------
# CALCE reader
# ---------------------------------------------------------------------------


def _parse_first_datetime(raw: pd.DataFrame, filepath: Path) -> pd.Timestamp:
    """Return the first Date_Time in the file as a Timestamp for chronological sorting.

    CALCE filenames use an unpadded M_D_YY scheme (e.g. CS2_35_10_15_10.xlsx) that
    does NOT sort lexicographically by date.  The Date_Time column inside each file is
    authoritative; the filename date is used as a fallback if the column is absent or
    unparseable.  Always returns a valid Timestamp so sorting never raises.
    """
    if "Date_Time" in raw.columns and len(raw) > 0:
        val = raw["Date_Time"].iloc[0]
        try:
            ts = pd.to_datetime(val)
            if not pd.isna(ts):
                return ts
        except Exception:
            pass
    # Fallback: parse M_D_YY from the filename stem.
    try:
        parts = filepath.stem.split("_")
        m, d, yy = int(parts[-3]), int(parts[-2]), int(parts[-1])
        return pd.Timestamp(year=2000 + yy, month=m, day=d)
    except Exception:
        pass
    return pd.Timestamp(0)  # epoch as last resort


def read_calce_cell_dir(cell_dir: Path, ambient_c: float) -> pd.DataFrame:
    """Read all Arbin .xls/.xlsx files in a CALCE cell directory and concatenate.

    CALCE distributes one Arbin file per test session; a single cell may have
    several files. Each file's Cycle_Index restarts at 0 — this function offsets
    subsequent files so the returned DataFrame has a globally unique cycle_index.
    Test_Time(s) is similarly offset for continuity.

    Files are concatenated in **chronological order** using the first Date_Time
    value recorded inside each file (authoritative; filenames use an unpadded
    M_D_YY scheme that does not sort lexicographically by date). If Date_Time
    cannot be parsed, the filename date is used as a fallback.

    Temperature is not logged by CALCE; a constant column = ambient_c is added.

    Parameters
    ----------
    cell_dir  : directory containing the .xls/.xlsx files for one cell
    ambient_c : stated ambient temperature (°C) — 25.0 for CALCE CS2/CX2

    Returns
    -------
    DataFrame with columns: cycle_index, time_s, current_a, voltage_v,
                            temperature_c, discharge_cap_ah
    """
    files = sorted(cell_dir.glob("*.xls")) + sorted(cell_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xls/.xlsx files in {cell_dir}")

    cm = CALCE_ARBIN_COLMAP
    needed = [
        cm["cycle_index"],
        cm["time_s"],
        cm["current_a"],
        cm["voltage_v"],
        cm["discharge_cap_ah"],
    ]

    # ------------------------------------------------------------------
    # Pass 1 — load every file and record its start timestamp for sorting
    # ------------------------------------------------------------------
    loaded: list[tuple[pd.Timestamp, pd.DataFrame]] = []

    for f in files:
        try:
            # CALCE xlsx files have two sheets: "Info" (metadata) and "Channel_X-XXX"
            # (data). Pick the Channel sheet by name; fall back to index 1.
            # fastexcel (Rust/calamine) is ~15× faster than openpyxl for this.
            wb = fastexcel.read_excel(str(f))
            sheet_idx = next(
                (
                    i
                    for i, s in enumerate(wb.sheet_names)
                    if s.lower().startswith("channel")
                ),
                1,
            )
            raw = wb.load_sheet_by_idx(sheet_idx).to_pandas()
        except Exception as exc:
            print(f"    Warning: cannot read {f.name}: {exc} — skipping")
            continue

        raw.columns = [str(c).strip() for c in raw.columns]

        missing = [c for c in needed if c not in raw.columns]
        if missing:
            print(f"    Warning: {f.name} missing {missing} — skipping")
            continue

        # Drop non-numeric rows (some Arbin files embed summary rows mid-sheet)
        raw = raw[pd.to_numeric(raw[cm["cycle_index"]], errors="coerce").notna()].copy()
        raw[cm["cycle_index"]] = raw[cm["cycle_index"]].astype(int)
        raw[cm["time_s"]] = raw[cm["time_s"]].astype(float)

        ts = _parse_first_datetime(raw, f)
        loaded.append((ts, raw))

    if not loaded:
        raise ValueError(f"No valid Arbin files in {cell_dir}")

    # Sort chronologically before applying cycle/time offsets
    loaded.sort(key=lambda x: x[0])

    # ------------------------------------------------------------------
    # Pass 2 — apply offsets in chronological order and concatenate
    # ------------------------------------------------------------------
    frames: list[pd.DataFrame] = []
    file_last_cycles: set[int] = set()
    cycle_offset = 0
    time_offset = 0.0

    for _ts, raw in loaded:
        raw = raw.copy()
        raw[cm["cycle_index"]] += cycle_offset
        raw[cm["time_s"]] += time_offset

        cycle_offset = int(raw[cm["cycle_index"]].max()) + 1
        time_offset = float(raw[cm["time_s"]].max()) + 1.0

        # Track the last cycle of each file — its discharge capacity is truncated
        # because the test session ended before the discharge completed.
        file_last_cycles.add(cycle_offset - 1)

        frames.append(raw)

    combined = pd.concat(frames, ignore_index=True)

    out = pd.DataFrame({
        "cycle_index": combined[cm["cycle_index"]].astype(int),
        "time_s": combined[cm["time_s"]].astype(float),
        "current_a": combined[cm["current_a"]].astype(float),
        "voltage_v": combined[cm["voltage_v"]].astype(float),
        "discharge_cap_ah": combined[cm["discharge_cap_ah"]].astype(float),
    })
    out["temperature_c"] = float(ambient_c)
    out["_file_last_cycle"] = out["cycle_index"].isin(file_last_cycles)
    return out


# ---------------------------------------------------------------------------
# BatteryArchive reader
# ---------------------------------------------------------------------------


def read_batteryarchive_csv(path: Path) -> pd.DataFrame:
    """Read a BatteryArchive *_timeseries.csv file.

    Returns standardised DataFrame with columns:
        cycle_index, time_s, current_a, voltage_v, temperature_c, discharge_cap_ah
    """
    cm = BATTERYARCHIVE_COLMAP
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]

    missing = [cm[k] for k in cm if cm[k] not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    return pd.DataFrame({
        "cycle_index": raw[cm["cycle_index"]].astype(int),
        "time_s": raw[cm["time_s"]].astype(float),
        "current_a": raw[cm["current_a"]].astype(float),
        "voltage_v": raw[cm["voltage_v"]].astype(float),
        "temperature_c": raw[cm["temperature_c"]].astype(float),
        "discharge_cap_ah": raw[cm["discharge_cap_ah"]].astype(float),
    })


# ---------------------------------------------------------------------------
# Generic voltage-grid extractor
# ---------------------------------------------------------------------------


def extract_timeseries_vg(
    df: pd.DataFrame,
    bid: str,
    c_nominal: float,
    ambient_c: float,
    has_temp: bool,
    v_lo: float = V_LO,
    v_hi: float = V_HI,
    n_grid: int = N_GRID,
    i_min_crate: float = 0.3,
    v_phys_max: float = 4.3,
    min_pts: int = 10,
    cap_lo_crate: float = 0.2,
    cap_hi_crate: float = 1.5,
) -> list[dict]:
    """Extract voltage-grid tensors from a long-format timeseries DataFrame.

    Reused for CALCE (Arbin .xls) and BatteryArchive (_timeseries.csv) cells.
    Expects df with standardised columns from read_calce_cell_dir /
    read_batteryarchive_csv: cycle_index, time_s, current_a, voltage_v,
    temperature_c, discharge_cap_ah.

    For each cycle:
    1. Isolates the positive-current (charging) rows.
    2. Calls resample_voltage_grid (voltage_grid.py — not modified) on the CC segment.
    3. Normalises the |I| channel to C-rate in-place: tensor[:, 0] /= c_nominal.
    4. Reads discharge capacity = max - min of discharge_cap_ah in negative-current rows
       (delta handles both per-cycle-reset and cumulative-from-test-start Arbin formats).

    Parameters
    ----------
    df          : standardised timeseries for one cell
    bid         : battery identifier string
    c_nominal   : rated capacity (Ah) — used to set i_min in Amps and to normalise |I|
    ambient_c   : imputed temperature (°C) when has_temp=False (ignored otherwise)
    has_temp    : if False, overwrite the temperature_c column with ambient_c
    i_min_crate : minimum charge current for CC onset detection, in C-rate units.
                  Converted to Amps as i_min_crate * c_nominal before being passed
                  to resample_voltage_grid (which stays in Amp units).
    cap_lo_crate : cycles whose discharge delta < cap_lo_crate * c_nominal are flagged
                  as NaN (partial / characterisation discharges, not a full 1C cycle).
    cap_hi_crate : cycles whose discharge delta > cap_hi_crate * c_nominal are flagged
                  as NaN (malformed Arbin Cycle_Index aggregating multiple sub-cycles).
                  Measured across all CALCE cells: only 1.3% of cycles are affected.

    Returns
    -------
    list of dicts with keys:
        battery_id, cycle_index, tensor (ndarray|None), mask (ndarray|None),
        coverage, cc_dt, cc_slope, capacity_ahr
    """
    if not has_temp:
        df = df.copy()
        df["temperature_c"] = float(ambient_c)

    i_min_amps = i_min_crate * c_nominal  # threshold in Amps for resample_voltage_grid
    has_file_last = "_file_last_cycle" in df.columns

    records: list[dict] = []

    for cycle_idx, grp in df.groupby("cycle_index"):
        grp = grp.sort_values("time_s").reset_index(drop=True)

        # File-last cycles have a truncated discharge (session ended before the
        # discharge completed).  Discharge_Capacity only records the partial amount,
        # so the label is unreliable regardless of the delta value.
        if has_file_last and bool(grp["_file_last_cycle"].any()):
            cap_ahr = float("nan")
        else:
            # Discharge capacity: max delta across contiguous negative-current runs.
            # Split on runs rather than taking a single (max - min) over the whole cycle
            # because RPT characterisation cycles interleave multiple discharges at
            # different C-rates within one Arbin Cycle_Index — a naive max-min would sum
            # all of them.  Each contiguous run is one discrete discharge step; we take
            # the largest, which corresponds to the most complete (slowest-rate) discharge.
            dis_rows = grp[grp["current_a"] < -0.01]
            if len(dis_rows) > 0:
                idx = dis_rows.index.to_numpy()
                run_id = np.concatenate([[0], np.cumsum(np.diff(idx) > 1)])
                run_caps = [
                    float(
                        dis_rows.iloc[run_id == r]["discharge_cap_ah"].max()
                        - dis_rows.iloc[run_id == r]["discharge_cap_ah"].min()
                    )
                    for r in np.unique(run_id)
                ]
                cap_ahr = float(max(run_caps))
            else:
                cap_ahr = float("nan")

            # Reject implausible capacities: below-band = partial/characterisation
            # discharge (not a full cycle); above-band = malformed Arbin Cycle_Index
            # that aggregates multiple sub-cycles into one row group.
            if not np.isnan(cap_ahr):
                if (
                    cap_ahr < cap_lo_crate * c_nominal
                    or cap_ahr > cap_hi_crate * c_nominal
                ):
                    cap_ahr = float("nan")

        tensor, mask_arr = None, None
        cc_sc = {"cc_dt": 0.0, "cc_slope": 0.0, "coverage": 0.0}

        charge_rows = grp[grp["current_a"] > 0.0]
        if len(charge_rows) >= 2:
            V = charge_rows["voltage_v"].to_numpy(dtype=float)
            I = charge_rows["current_a"].to_numpy(dtype=float)  # positive
            Temp = charge_rows["temperature_c"].to_numpy(dtype=float)
            Time = charge_rows["time_s"].to_numpy(dtype=float)

            tensor, mask_arr, cc_sc = resample_voltage_grid(
                V,
                I,
                Temp,
                Time,
                v_lo=v_lo,
                v_hi=v_hi,
                n_grid=n_grid,
                i_min=i_min_amps,
                v_phys_max=v_phys_max,
                min_pts=min_pts,
            )

            if tensor is not None:
                tensor = tensor.copy()
                tensor[:, 0] /= c_nominal  # |I| → C-rate

        records.append({
            "battery_id": bid,
            "cycle_index": int(cycle_idx),
            "tensor": tensor,
            "mask": mask_arr,
            "capacity_ahr": cap_ahr,
            **cc_sc,
        })

    return records
