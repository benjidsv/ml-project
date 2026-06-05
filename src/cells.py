"""Cell registry for the extended multi-dataset voltage-grid pipeline.

Each CellSpec encodes everything the build pipeline needs to know about a cell:
where to find the raw data, what loader to use, its nominal capacity (for C-rate
normalisation and target denominator), and its dataset group (for LODO).

Usage
-----
    from cells import CELLS, EXCLUDED, cells_by_tier, get_cell

    tier1 = cells_by_tier(1)   # NASA + CALCE
    tier2 = cells_by_tier(2)   # BatteryArchive (add after download)

Extending
---------
    To add new cells (e.g. CALCE CX2 or BatteryArchive Tier 2), append CellSpec
    entries to CELLS below.  Add a corresponding entry to EXCLUDED for any cell
    that was downloaded but deliberately skipped, so the decision is documented.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CellSpec:
    bid: str             # unique battery identifier (used as battery_id in DataFrames)
    chemistry: str       # e.g. "LCO", "NMC-LCO", "NCA", "NMC"
    c_nominal_ah: float  # rated capacity (Ah) from cell spec sheet
    ambient_c: float     # ambient temperature (°C) for T-channel imputation
    has_temp: bool       # True = per-sample T in raw data; False = impute ambient_c
    group: str           # LODO fold key — one string per dataset source
    tier: int            # 1 = same chemistry/voltage; 2 = different chemistry, same V_max
    rel_path: str        # path relative to project root (.mat for NASA; dir for CALCE/BA)
    loader: str          # dispatch key: "nasa_ctrl"|"nasa_rw"|"calce_arbin"|"batteryarchive"


# ---------------------------------------------------------------------------
# Tier 1 — NASA PCoE (existing, unchanged loaders)
# ---------------------------------------------------------------------------

_NASA_CTRL = [
    CellSpec(
        bid="B0005", chemistry="LCO", c_nominal_ah=2.0, ambient_c=24.0, has_temp=True,
        group="nasa", tier=1,
        rel_path="data/controlled/1. BatteryAgingARC-FY08Q4/B0005.mat",
        loader="nasa_ctrl",
    ),
    CellSpec(
        bid="B0006", chemistry="LCO", c_nominal_ah=2.0, ambient_c=24.0, has_temp=True,
        group="nasa", tier=1,
        rel_path="data/controlled/1. BatteryAgingARC-FY08Q4/B0006.mat",
        loader="nasa_ctrl",
    ),
    CellSpec(
        bid="B0007", chemistry="LCO", c_nominal_ah=2.0, ambient_c=24.0, has_temp=True,
        group="nasa", tier=1,
        rel_path="data/controlled/1. BatteryAgingARC-FY08Q4/B0007.mat",
        loader="nasa_ctrl",
    ),
    CellSpec(
        bid="B0018", chemistry="LCO", c_nominal_ah=2.0, ambient_c=24.0, has_temp=True,
        group="nasa", tier=1,
        rel_path="data/controlled/1. BatteryAgingARC-FY08Q4/B0018.mat",
        loader="nasa_ctrl",
    ),
]

# RW9 is intentionally absent — see EXCLUDED below.
_NASA_RW = [
    CellSpec(
        bid="RW1", chemistry="LCO", c_nominal_ah=2.2, ambient_c=24.0, has_temp=True,
        group="nasa", tier=1,
        rel_path="data/randomized/Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW1.mat",
        loader="nasa_rw",
    ),
    CellSpec(
        bid="RW13", chemistry="LCO", c_nominal_ah=2.2, ambient_c=24.0, has_temp=True,
        group="nasa", tier=1,
        rel_path="data/randomized/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW13.mat",
        loader="nasa_rw",
    ),
    CellSpec(
        bid="RW14", chemistry="LCO", c_nominal_ah=2.2, ambient_c=24.0, has_temp=True,
        group="nasa", tier=1,
        rel_path="data/randomized/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW14.mat",
        loader="nasa_rw",
    ),
    CellSpec(
        bid="RW15", chemistry="LCO", c_nominal_ah=2.2, ambient_c=24.0, has_temp=True,
        group="nasa", tier=1,
        rel_path="data/randomized/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW15.mat",
        loader="nasa_rw",
    ),
    CellSpec(
        bid="RW16", chemistry="LCO", c_nominal_ah=2.2, ambient_c=24.0, has_temp=True,
        group="nasa", tier=1,
        rel_path="data/randomized/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW16.mat",
        loader="nasa_rw",
    ),
    CellSpec(
        bid="RW17", chemistry="LCO", c_nominal_ah=2.2, ambient_c=24.0, has_temp=True,
        group="nasa", tier=1,
        rel_path="data/randomized/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW17.mat",
        loader="nasa_rw",
    ),
    CellSpec(
        bid="RW19", chemistry="LCO", c_nominal_ah=2.2, ambient_c=24.0, has_temp=True,
        group="nasa", tier=1,
        rel_path="data/randomized/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW19.mat",
        loader="nasa_rw",
    ),
    CellSpec(
        bid="RW20", chemistry="LCO", c_nominal_ah=2.2, ambient_c=24.0, has_temp=True,
        group="nasa", tier=1,
        rel_path="data/randomized/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW20.mat",
        loader="nasa_rw",
    ),
]

# ---------------------------------------------------------------------------
# Tier 1 — CALCE CS2 (standard CC-CV, Arbin tester only)
# ---------------------------------------------------------------------------
# Paths: data/calce/{bid}/ — flat structure, one dir per cell.
# C_nominal = 1.1 Ah (LCO prismatic); ambient = 25 °C (CALCE spec).
# CS2_3 and CS2_9 are excluded — see EXCLUDED below.

_CALCE_CS2 = [
    CellSpec(
        bid="CS2_33", chemistry="LCO", c_nominal_ah=1.1, ambient_c=25.0, has_temp=False,
        group="calce", tier=1,
        rel_path="data/calce/CS2_33",
        loader="calce_arbin",
    ),
    CellSpec(
        bid="CS2_34", chemistry="LCO", c_nominal_ah=1.1, ambient_c=25.0, has_temp=False,
        group="calce", tier=1,
        rel_path="data/calce/CS2_34",
        loader="calce_arbin",
    ),
    CellSpec(
        bid="CS2_35", chemistry="LCO", c_nominal_ah=1.1, ambient_c=25.0, has_temp=False,
        group="calce", tier=1,
        rel_path="data/calce/CS2_35",
        loader="calce_arbin",
    ),
    CellSpec(
        bid="CS2_36", chemistry="LCO", c_nominal_ah=1.1, ambient_c=25.0, has_temp=False,
        group="calce", tier=1,
        rel_path="data/calce/CS2_36",
        loader="calce_arbin",
    ),
    CellSpec(
        bid="CS2_37", chemistry="LCO", c_nominal_ah=1.1, ambient_c=25.0, has_temp=False,
        group="calce", tier=1,
        rel_path="data/calce/CS2_37",
        loader="calce_arbin",
    ),
    CellSpec(
        bid="CS2_38", chemistry="LCO", c_nominal_ah=1.1, ambient_c=25.0, has_temp=False,
        group="calce", tier=1,
        rel_path="data/calce/CS2_38",
        loader="calce_arbin",
    ),
]

# ---------------------------------------------------------------------------
# Tier 1 — CALCE CX2 (standard CC-CV, Arbin tester)
# ---------------------------------------------------------------------------
# Confirmed cells from downloaded data: CX2_16, CX2_33, CX2_35, CX2_37, CX2_38.
# C_nominal = 1.1 Ah (same LCO prismatic chemistry as CS2).

_CALCE_CX2 = [
    CellSpec(
        bid="CX2_33", chemistry="LCO", c_nominal_ah=1.1, ambient_c=25.0, has_temp=False,
        group="calce", tier=1,
        rel_path="data/calce/CX2_33",
        loader="calce_arbin",
    ),
    CellSpec(
        bid="CX2_35", chemistry="LCO", c_nominal_ah=1.1, ambient_c=25.0, has_temp=False,
        group="calce", tier=1,
        rel_path="data/calce/CX2_35",
        loader="calce_arbin",
    ),
    CellSpec(
        bid="CX2_37", chemistry="LCO", c_nominal_ah=1.1, ambient_c=25.0, has_temp=False,
        group="calce", tier=1,
        rel_path="data/calce/CX2_37",
        loader="calce_arbin",
    ),
    CellSpec(
        bid="CX2_38", chemistry="LCO", c_nominal_ah=1.1, ambient_c=25.0, has_temp=False,
        group="calce", tier=1,
        rel_path="data/calce/CX2_38",
        loader="calce_arbin",
    ),
]


# ---------------------------------------------------------------------------
# Tier 2 — BatteryArchive (add after download)
# ---------------------------------------------------------------------------
# TODO: add HNEI, SNL (NCA+NMC only, drop LFP), UNIBO entries here.
# All use loader="batteryarchive".
# rel_path = path to the cell's *_timeseries.csv file.
_BATTERYARCHIVE_TIER2: list[CellSpec] = []  # <-- populate after download


# ---------------------------------------------------------------------------
# Master list — used by build_extended_dataset when no cell_specs are passed
# ---------------------------------------------------------------------------

CELLS: list[CellSpec] = (
    _NASA_CTRL
    + _NASA_RW
    + _CALCE_CS2
    + _CALCE_CX2
    + _BATTERYARCHIVE_TIER2
)


# ---------------------------------------------------------------------------
# Documented exclusions
# ---------------------------------------------------------------------------
# Keys are battery/group identifiers; values are the reason for exclusion.
# Update this dict whenever a cell is considered and dropped.

EXCLUDED: dict[str, str] = {
    # NASA randomized
    "RW9": (
        "Belongs to the random-SOC group RW9–RW12; these batteries start at arbitrary SOH. "
        "Papers drop the entire group. We had mistakenly kept only RW9."
    ),
    "RW2": "Sensor errors — current/voltage traces are corrupted.",
    "RW18": "Sensor errors — current/voltage traces are corrupted.",
    # NASA controlled — excluded groups
    "B0025–B0028 (hot group, 43 °C)": (
        "Cycled at 43 °C; these cells never reach the 80 % capacity EOL criterion. "
        "Cannot compute a finite RUL."
    ),
    "B0029–B0044 (cold/mixed groups, 4 °C)": (
        "4 °C testing with inconsistent mixed-load protocols; "
        "SOH trajectories are non-monotone and incomparable to room-temp cells."
    ),
    "B0041": (
        "Anomalous starting capacity (well below the group baseline); "
        "likely a defective cell."
    ),
    "B0045–B0056": "Additional controlled groups not retained (see README).",
    # CALCE CS2
    "CS2_8": "Tested on CADEX cycler — incompatible column format.",
    "CS2_21": "Tested on CADEX cycler — incompatible column format.",
    "CS2_3": (
        "Variable discharge protocol: each cycle alternates discharge current among "
        "0.11, 0.22, 0.55, 1.1, 1.65, 2.2 A. Measured capacity per cycle depends on "
        "the C-rate used that cycle, making the label noisy and incomparable across cycles."
    ),
    "CS2_9": (
        "Variable discharge protocol: each cycle alternates discharge current among "
        "0.11, 0.22, 0.55, 1.1, 1.65, 2.2 A. Same reason as CS2_3."
    ),
    # CALCE CX2 — cells not downloaded / excluded by protocol
    "CX2 — CADEX-tester cells": (
        "Cells tested on CADEX cycler have different column names; "
        "incompatible with the Arbin reader."
    ),
    "CX2 — varying-temperature cells (25–55 °C)": (
        "Multi-temperature testing without temperature as a model input creates "
        "unlearnable variance between cells."
    ),
    "CX2_16": (
        "Hard protocol change at cycle 1152: first 1114 cycles discharge at −0.55 A (0.5C), "
        "remaining 894 cycles at −0.675 A (0.61C). The two phases measure different physical "
        "quantities — capacity at different C-rates is not comparable on a single SOH scale. "
        "The step discontinuity at cycle 1152 is a test-protocol artifact, not degradation."
    ),
    "CX2 — variable-discharge-rate cells": (
        "Discharge current alternated each cycle (same protocol as CS2_3/CS2_9). "
        "Capacity label is C-rate-dependent and non-comparable across cycles."
    ),
    "CX2 — partial-cycling cells": (
        "Partial cycling protocol — unusual discharge behaviour."
    ),
    # BatteryArchive
    "SNL LFP cells": (
        "LFP chemistry has a 3.6 V voltage plateau — outside the 3.8–4.2 V window. "
        "Drop these cells before building the dataset."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cells_by_tier(tier: int) -> list[CellSpec]:
    """Return all cells with the given tier number."""
    return [c for c in CELLS if c.tier == tier]


def get_cell(bid: str) -> CellSpec:
    """Look up a cell by its battery_id; raises KeyError if not found."""
    for c in CELLS:
        if c.bid == bid:
            return c
    raise KeyError(f"No cell with bid={bid!r} in the registry. "
                   f"Check EXCLUDED for a documented reason, or add a new CellSpec.")
