"""Multi-dataset voltage-grid pipeline — extended functions.

This module adds new functionality for the multi-source dataset without touching
src/voltage_grid.py.  All existing NASA-only code remains unchanged and callable
as before.

New public API
--------------
    build_extended_dataset   registry-driven builder (NASA + CALCE + BatteryArchive)
    attach_targets_extended  per-cell C_nom normalisation; returns ds_groups for LODO
    save_npz_ext             save extended .npz (adds ds_groups array)
    load_npz_ext             load extended .npz
    run_grouped_cv           GroupKFold-8 / LODO runner for DL models; mirrors run_lobo API
    vg_scalar_features       engineer charge-curve scalar descriptors for RF baseline
    run_rf_grouped_cv        GroupKFold-8 / LODO runner for sklearn Random Forest
    per_fold_scatter_ext     per-fold scatter plot (uses held_bids/held_group keys)
    per_fold_loss_curves_ext per-fold loss curves for DL models (skips RF folds)

Reused unchanged from voltage_grid.py
--------------------------------------
    resample_voltage_grid, extract_controlled_vg, extract_randomized_vg
    global_scale, _reapply_fill
    lobo_metrics, aggregate
    per_fold_scatter, per_fold_loss_curves   (NASA-only; use *_ext variants here)
    V_LO, V_HI, N_GRID

Reused unchanged from vg_models.py / sequence.py
-------------------------------------------------
    vg_models.predict_vg, sequence.train_model
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# ── reuse from existing modules (not modified) ───────────────────────────────
from voltage_grid import (
    V_LO,
    V_HI,
    N_GRID,
    extract_controlled_vg,
    extract_randomized_vg,
    global_scale,
    _reapply_fill,
    lobo_metrics,
    aggregate,
    per_fold_scatter,
    per_fold_loss_curves,
)

from cells import CELLS, CellSpec
from extract_generic import (
    extract_timeseries_vg,
    read_calce_cell_dir,
    read_batteryarchive_csv,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_nasa_labels(project_root: Path) -> pd.DataFrame:
    """Load capacity_ahr labels from the NASA processed CSVs."""
    ctrl = pd.read_csv(
        project_root / "data/processed/controlled.csv",
        usecols=["battery_id", "cycle_index", "capacity_ahr"],
    )
    rw = pd.read_csv(
        project_root / "data/processed/randomized.csv",
        usecols=["battery_id", "cycle_index", "capacity_ahr"],
    )
    return pd.concat([ctrl, rw], ignore_index=True)


def _normalize_crate(records: list[dict], c_nominal: float) -> list[dict]:
    """Divide the |I| channel (index 0) of each tensor by c_nominal in-place copy.

    NASA extractors return raw Amps; this converts to C-rate to match the
    CALCE/BatteryArchive tensors produced by extract_timeseries_vg.
    """
    for r in records:
        if r["tensor"] is not None:
            t = r["tensor"].copy()
            t[:, 0] /= c_nominal
            r["tensor"] = t
    return records


def _drop_partial_charge_dips(
    recs: list[dict],
    window: int = 5,
    threshold: float = 0.9,
    max_drop_ah: float = float("inf"),
) -> list[dict]:
    """Null capacity_ahr for cycles discharged from a partial state of charge.

    Some CALCE test protocols periodically skip the CV taper phase (step 4 in
    Arbin logs), discharging the cell before it is fully recharged.  The result
    is an isolated capacity dip — the surrounding cycles return to the true level.

    Detection: for each valid cycle, if capacity_ahr < threshold * rolling_max
    over a centred ±window-cycle band, the discharge started from a lower SOC
    than usual and the label is unreliable.  A centred window means even a single
    post-dip recovery cycle is enough to establish the true-capacity reference,
    so genuine end-of-life degradation (which does not recover) is not flagged.

    max_drop_ah caps how deep a flagged dip may be (not used for NASA cells,
    which skip this filter entirely).
    """
    valid_recs = sorted(
        [r for r in recs if not np.isnan(r.get("capacity_ahr", float("nan")))],
        key=lambda r: r["cycle_index"],
    )
    if len(valid_recs) < 3:
        return recs

    cap_arr = np.array([r["capacity_ahr"] for r in valid_recs])
    rolling_max = (
        pd
        .Series(cap_arr)
        .rolling(window=2 * window + 1, center=True, min_periods=1)
        .max()
        .to_numpy()
    )
    flagged = {
        valid_recs[i]["cycle_index"]
        for i in range(len(cap_arr))
        if (
            cap_arr[i] < threshold * rolling_max[i]
            and rolling_max[i] - cap_arr[i] <= max_drop_ah
        )
    }
    for r in recs:
        if r["cycle_index"] in flagged:
            r["capacity_ahr"] = float("nan")
    return recs


def _attach_nasa_labels(
    records: list[dict], bid: str, labels: pd.DataFrame
) -> list[dict]:
    """Join capacity_ahr from the NASA processed-CSV label table."""
    lmap = (
        labels[labels["battery_id"] == bid]
        .set_index("cycle_index")["capacity_ahr"]
        .to_dict()
    )
    for r in records:
        r["capacity_ahr"] = lmap.get(r["cycle_index"], float("nan"))
    return records


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_extended_dataset(
    project_root: Path,
    cell_specs: list[CellSpec] | None = None,
    v_lo: float = V_LO,
    v_hi: float = V_HI,
    n_grid: int = N_GRID,
    i_min_crate: float = 0.3,
    v_phys_max: float = 4.3,
    min_pts: int = 10,
) -> pd.DataFrame:
    """Extract voltage-grid tensors for all specified cells.

    Differs from build_voltage_grid_dataset (voltage_grid.py) in that it:
    - Is driven by the cell registry (cells.py)
    - Normalises the |I| channel to C-rate per cell (|I| / C_nom)
    - Imputes T = ambient_c for cells without per-sample temperature
    - Attaches capacity_ahr to every row (from processed CSVs for NASA;
      from cycler discharge capacity for CALCE / BatteryArchive)
    - Attaches c_nominal_ah and ds_group for downstream normalisation + LODO

    Parameters
    ----------
    project_root : Path  repository root (parent of data/, src/)
    cell_specs   : cells to process; defaults to cells.CELLS (all registered cells)

    Returns
    -------
    DataFrame with columns:
        battery_id, cycle_index, tensor (ndarray|None), mask (ndarray|None),
        coverage, cc_dt, cc_slope, capacity_ahr, c_nominal_ah, ds_group
    """
    from battery import load_controlled, load_randomized

    specs = cell_specs if cell_specs is not None else CELLS

    # Pre-load NASA labels once if any NASA cell is in the spec list
    nasa_labels: pd.DataFrame | None = None
    if any(s.loader in ("nasa_ctrl", "nasa_rw") for s in specs):
        nasa_labels = _load_nasa_labels(project_root)

    all_records: list[dict] = []

    for spec in specs:
        print(f"  {spec.bid}  [{spec.group}  {spec.loader}] ...", end="  ", flush=True)
        path = project_root / spec.rel_path

        if spec.loader == "nasa_ctrl":
            cycles = load_controlled(path)
            recs = extract_controlled_vg(
                cycles,
                spec.bid,
                v_lo=v_lo,
                v_hi=v_hi,
                n_grid=n_grid,
                i_min=i_min_crate * spec.c_nominal_ah,
                v_phys_max=v_phys_max,
                min_pts=min_pts,
            )
            # cycle_index 0 is a formation/conditioning pass with a step-down CC
            # protocol (current ~1.24C tapering to 0.75C).  All subsequent cycles
            # use standard 0.75C CC-CV.  Drop the formation cycle so the model
            # never sees a fundamentally different charge-curve shape.
            recs = [r for r in recs if r["cycle_index"] != 0]
            recs = _normalize_crate(recs, spec.c_nominal_ah)
            recs = _attach_nasa_labels(recs, spec.bid, nasa_labels)

        elif spec.loader == "nasa_rw":
            steps = load_randomized(path)
            recs = extract_randomized_vg(
                steps,
                spec.bid,
                v_lo=v_lo,
                v_hi=v_hi,
                n_grid=n_grid,
                i_min=i_min_crate * spec.c_nominal_ah,
                v_phys_max=v_phys_max,
                min_pts=min_pts,
            )
            recs = _normalize_crate(recs, spec.c_nominal_ah)
            recs = _attach_nasa_labels(recs, spec.bid, nasa_labels)

        elif spec.loader == "calce_arbin":
            df = read_calce_cell_dir(path, spec.ambient_c)
            recs = extract_timeseries_vg(
                df,
                spec.bid,
                c_nominal=spec.c_nominal_ah,
                ambient_c=spec.ambient_c,
                has_temp=spec.has_temp,
                v_lo=v_lo,
                v_hi=v_hi,
                n_grid=n_grid,
                i_min_crate=i_min_crate,
                v_phys_max=v_phys_max,
                min_pts=min_pts,
            )

        elif spec.loader == "batteryarchive":
            df = read_batteryarchive_csv(path)
            recs = extract_timeseries_vg(
                df,
                spec.bid,
                c_nominal=spec.c_nominal_ah,
                ambient_c=spec.ambient_c,
                has_temp=spec.has_temp,
                v_lo=v_lo,
                v_hi=v_hi,
                n_grid=n_grid,
                i_min_crate=i_min_crate,
                v_phys_max=v_phys_max,
                min_pts=min_pts,
            )

        else:
            raise ValueError(f"Unknown loader {spec.loader!r} for {spec.bid}")

        if spec.loader not in ("nasa_ctrl", "nasa_rw"):
            recs = _drop_partial_charge_dips(recs)

        for r in recs:
            r["c_nominal_ah"] = spec.c_nominal_ah
            r["ds_group"] = spec.group

        valid = sum(1 for r in recs if r["tensor"] is not None)
        mean_cov = np.mean([r["coverage"] for r in recs]) if recs else 0.0
        print(f"{len(recs)} cycles, {valid} valid, mean_cov={mean_cov:.3f}")
        all_records.extend(recs)

    return pd.DataFrame(all_records)


# ---------------------------------------------------------------------------
# Target attachment
# ---------------------------------------------------------------------------


def attach_targets_extended(
    curve_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack arrays from curve_df for model training.

    Expects curve_df produced by build_extended_dataset, which already contains
    capacity_ahr and c_nominal_ah for every row.  This function drops rows with
    missing tensors or labels, then normalises capacity to the per-cell rated value.

    Returns
    -------
    X         : (n, n_grid, 3) float32 — C-rate-normalised tensors
    mask      : (n, n_grid) bool
    y         : (n,) float32 — capacity_ahr / c_nominal_ah
    groups    : (n,) object  — battery_id strings
    cidx      : (n,) int64   — cycle_index
    ds_groups : (n,) object  — dataset group label (for LODO splitter)
    """
    df = curve_df.copy()
    df = df[df["tensor"].apply(lambda t: t is not None)]
    df = df[df["capacity_ahr"].notna()]
    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid samples after filtering — check extraction output.")

    n_grid = df["tensor"].iloc[0].shape[0]

    X = np.stack(df["tensor"].values).astype(np.float32)
    mask_arr = np.stack(df["mask"].values)
    y = (df["capacity_ahr"] / df["c_nominal_ah"]).to_numpy(dtype=np.float32)
    groups = df["battery_id"].to_numpy()
    cidx = df["cycle_index"].to_numpy(dtype=np.int64)
    ds_groups = df["ds_group"].to_numpy()

    print(
        f"  Dataset: {len(df)} samples, {n_grid} grid pts, "
        f"y ∈ [{y.min():.3f}, {y.max():.3f}], "
        f"{len(set(groups))} cells, {len(set(ds_groups))} dataset groups"
    )
    return X, mask_arr, y, groups, cidx, ds_groups


# ---------------------------------------------------------------------------
# NPZ cache (extended — includes ds_groups)
# ---------------------------------------------------------------------------


def save_npz_ext(
    path: Path,
    X: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cidx: np.ndarray,
    ds_groups: np.ndarray,
    v_lo: float = V_LO,
    v_hi: float = V_HI,
    n_grid: int = N_GRID,
) -> None:
    """Save the extended dataset to a compressed .npz file."""
    np.savez_compressed(
        path,
        X=X,
        mask=mask,
        y=y,
        groups=groups,
        cidx=cidx,
        ds_groups=ds_groups,
        config=np.array([v_lo, v_hi, n_grid]),
    )
    print(f"  Saved to {path}")


def load_npz_ext(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load an extended .npz file produced by save_npz_ext.

    Returns (X, mask, y, groups, cidx, ds_groups).
    """
    d = np.load(path, allow_pickle=True)
    return d["X"], d["mask"], d["y"], d["groups"], d["cidx"], d["ds_groups"]


# ---------------------------------------------------------------------------
# Grouped cross-validation runner
# ---------------------------------------------------------------------------


def run_grouped_cv(
    model_factory: Callable,
    X: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cidx: np.ndarray,
    device,
    splitter,
    *,
    cv_groups: np.ndarray | None = None,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 50,
    scheduler: str | None = "cosine",
    progress: bool = True,
) -> list[dict]:
    """Grouped cross-validation for the extended multi-dataset pipeline.

    Mirrors run_lobo (voltage_grid.py) but generalises the fold iteration to any
    scikit-learn GroupSplit-compatible splitter.  Does not modify run_lobo.

    Parameters
    ----------
    model_factory : callable() → nn.Module
        Called fresh for each fold; same signature as run_lobo.
    X, mask, y, groups, cidx : arrays from attach_targets_extended
        groups = battery_id per sample (always used for inner-val logic).
    device : torch.device
    splitter : sklearn splitter with .split(X, groups=) and .get_n_splits()
        GroupKFold(n_splits=8)   — headline 8-fold whole-cell CV
        LeaveOneGroupOut()       — LODO; pass ds_groups as cv_groups
    cv_groups : optional array passed to splitter.split() as the groups argument.
        For GroupKFold: omit (defaults to battery groups — whole-cell folds).
        For LODO:       pass ds_groups (dataset-level labels from attach_targets_extended).
    epochs, batch_size, lr, weight_decay, patience, scheduler, progress
        Forwarded to sequence.train_model — same defaults as run_lobo.

    Returns
    -------
    list of fold dicts, each with:
        held_bids, held_group, val_bid,
        y_te, y_pred, cidx_te,
        train_curve, val_curve, best_epoch,
        metrics  (keys: mae, rmse, r2, skill, spearman, naive_mae, n)

    Example
    -------
        from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
        from vg_extended import run_grouped_cv, load_npz_ext
        from vg_models import VGCNNLSTM

        X, mask, y, groups, cidx, ds_groups = load_npz_ext(path)

        # Headline: 8-fold whole-cell CV
        folds_gkf = run_grouped_cv(
            lambda: VGCNNLSTM(), X, mask, y, groups, cidx, device,
            GroupKFold(n_splits=8),
        )

        # Transfer: Leave-One-Dataset-Out
        folds_lodo = run_grouped_cv(
            lambda: VGCNNLSTM(), X, mask, y, groups, cidx, device,
            LeaveOneGroupOut(), cv_groups=ds_groups,
        )
    """
    import torch
    from sequence import train_model
    from vg_models import predict_vg

    split_groups = cv_groups if cv_groups is not None else groups
    n_folds = splitter.get_n_splits(X, groups=split_groups)
    folds: list[dict] = []

    for fold_idx, (_, test_idx) in enumerate(splitter.split(X, groups=split_groups)):
        is_test = np.zeros(len(X), dtype=bool)
        is_test[test_idx] = True
        is_avail = ~is_test

        # Inner-val: last-alphabetical battery among available train cells.
        # Deterministic and dataset-agnostic (no CONTROLLED_BIDS dependency).
        avail_bids = sorted(set(groups[is_avail]))
        val_bid = avail_bids[-1]
        is_val = is_avail & (groups == val_bid)
        is_train = is_avail & ~is_val

        X_tr, mask_tr, y_tr = X[is_train], mask[is_train], y[is_train]
        X_val, mask_val, y_val = X[is_val], mask[is_val], y[is_val]
        X_te, mask_te, y_te = X[is_test], mask[is_test], y[is_test]

        X_tr_s, X_val_s, X_te_s = global_scale(X_tr, X_val, X_te, mask_tr)
        X_tr_s = _reapply_fill(X_tr_s, mask_tr)
        X_val_s = _reapply_fill(X_val_s, mask_val)
        X_te_s = _reapply_fill(X_te_s, mask_te)

        torch.manual_seed(fold_idx * 31 + 42)
        model = model_factory().to(device)

        model, train_curve, val_curve, best_ep = train_model(
            model,
            X_tr_s,
            mask_tr,
            y_tr,
            X_val_s,
            mask_val,
            y_val,
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

        held_bids = sorted(set(groups[is_test]))
        held_group = sorted(set(split_groups[is_test]))
        held_str = ",".join(held_group[:3]) + ("…" if len(held_group) > 3 else "")

        print(
            f"  [{fold_idx + 1:2d}/{n_folds}] held={held_str:20s}  "
            f"val={val_bid}  "
            f"MAE={metrics['mae']:.4f}  R²={metrics['r2']:.3f}  "
            f"best_ep={best_ep}"
        )

        folds.append({
            "held_bids": held_bids,
            "held_group": held_group,
            "val_bid": val_bid,
            "y_te": y_te,
            "y_pred": y_pred,
            "cidx_te": cidx[is_test],
            "train_curve": train_curve,
            "val_curve": val_curve,
            "best_epoch": best_ep,
            "metrics": metrics,
        })

    return folds


# ---------------------------------------------------------------------------
# Scalar feature engineering for the RF baseline
# ---------------------------------------------------------------------------


def vg_scalar_features(
    X: np.ndarray,
    mask: np.ndarray,
    *,
    v_lo: float = V_LO,
    v_hi: float = V_HI,
) -> tuple[np.ndarray, list[str]]:
    """Engineer scalar descriptors from the voltage-grid tensor for an RF baseline.

    All features are derived solely from the CC charging curve (X, mask) without
    accessing discharge capacity — no target leakage.

    Grid layout (descending): index 0 = V_HI (end of CC phase in time),
    index n_grid-1 = V_LO (start of CC phase).  Valid positions form a leading
    block at the start of the array (mask = [T…T F…F]).

    X    : (n, n_grid, 3) float32  channels: [C-rate, Temperature (°C), t_elapsed (s)]
    mask : (n, n_grid) bool        True = valid grid position

    Returns
    -------
    feat       : (n, 11) float32
    feat_names : list[str] of length 11

    Feature descriptions
    --------------------
    coverage   fraction of the 128-pt grid covered (leading-block length / n_grid)
    v_start    voltage at CC onset = grid V at last valid position (lowest covered V)
    cc_dt      CC phase duration (s) = t_elapsed at V_HI end (position 0)
    cc_slope   average voltage rise rate V/s  = (v_hi - v_start) / cc_dt
    crate_mean mean C-rate over valid positions
    crate_max  max  C-rate over valid positions
    crate_start C-rate at CC onset (last valid position = lowest covered V)
    temp_mean  mean temperature (°C) over valid positions
    temp_max   max  temperature (°C) over valid positions
    t_total    total CC duration (s) — same as cc_dt, alias kept for RF feature set
    r_proxy    charge-side ohmic proxy: (k * dV_step) / crate_start (k=5 grid steps).
               Mirrors battery.py dcir_proxy_* adapted to the charge curve:
               higher → lower onset C-rate → more resistive / aged cell.
    """
    N, n_grid, _ = X.shape
    vgrid = np.linspace(v_hi, v_lo, n_grid)  # descending (n_grid,)

    # Index of last valid position per sample (leading block: True at low indices)
    n_valid = mask.sum(axis=1).astype(int)
    n_valid = np.clip(n_valid, 1, n_grid)  # guard: at least 1 valid position
    last_vi = n_valid - 1  # (N,) index of lowest covered V

    sample_idx = np.arange(N)

    # coverage
    coverage = mask.mean(axis=1)  # (N,)

    # v_start: voltage at CC onset
    v_start = vgrid[last_vi]  # (N,) via fancy indexing

    # cc_dt / t_total: total CC duration — position 0 (V_HI) is always valid
    cc_dt = X[:, 0, 2].copy()  # (N,) t_elapsed at V_HI end
    t_total = cc_dt.copy()  # alias

    # cc_slope: average voltage rise rate (V/s)
    v_range = v_hi - v_start  # (N,) voltage span covered
    with np.errstate(invalid="ignore", divide="ignore"):
        cc_slope = np.where(cc_dt > 0, v_range / cc_dt, 0.0)  # (N,)

    # C-rate stats (channel 0) over valid positions only
    with np.errstate(invalid="ignore"):
        X_crate = np.where(mask, X[:, :, 0], np.nan)  # (N, n_grid)
        crate_mean = np.nanmean(X_crate, axis=1)  # (N,)
        crate_max = np.nanmax(X_crate, axis=1)  # (N,)
    crate_start = X[sample_idx, last_vi, 0]  # (N,) at onset

    # Temperature stats (channel 1) over valid positions only
    with np.errstate(invalid="ignore"):
        X_temp = np.where(mask, X[:, :, 1], np.nan)  # (N, n_grid)
        temp_mean = np.nanmean(X_temp, axis=1)  # (N,)
        temp_max = np.nanmax(X_temp, axis=1)  # (N,)

    # r_proxy: charge-side ohmic proxy
    # ΔV over k grid steps from onset / C-rate at onset.
    # The grid is uniform, so ΔV = k × dv_step is constant; r_proxy is therefore
    # proportional to 1/crate_start and approximates an ohmic slope — higher value
    # reflects lower onset C-rate, consistent with more resistive or degraded cells.
    k = 5
    dv_step = (v_hi - v_lo) / (n_grid - 1)
    eps = 1e-6
    r_proxy = (k * dv_step) / np.maximum(crate_start, eps)  # (N,)

    feat = np.column_stack([
        coverage,
        v_start,
        cc_dt,
        cc_slope,
        crate_mean,
        crate_max,
        crate_start,
        temp_mean,
        temp_max,
        t_total,
        r_proxy,
    ]).astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    feat_names = [
        "coverage",
        "v_start",
        "cc_dt",
        "cc_slope",
        "crate_mean",
        "crate_max",
        "crate_start",
        "temp_mean",
        "temp_max",
        "t_total",
        "r_proxy",
    ]
    return feat, feat_names


# ---------------------------------------------------------------------------
# RF grouped cross-validation runner
# ---------------------------------------------------------------------------


def run_rf_grouped_cv(
    feat: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cidx: np.ndarray,
    splitter,
    *,
    cv_groups: np.ndarray | None = None,
    n_estimators: int = 300,
    max_depth: int = 8,
    min_samples_leaf: int = 3,
    max_features: float = 0.6,
    random_state: int = 42,
    progress: bool = True,
) -> list[dict]:
    """Grouped cross-validation for a scikit-learn Random Forest baseline.

    Mirrors run_grouped_cv but for sklearn RF — no scaling, no inner-val fold,
    no train/val loss curves.  Returns fold dicts with the same keys as
    run_grouped_cv so aggregate() and per_fold_scatter_ext() work unchanged.
    DL-only keys (train_curve, val_curve, best_epoch, val_bid) are set to None.

    Parameters
    ----------
    feat       : (n, n_features) float32 — engineered scalar features, e.g. from
                 vg_scalar_features()
    y          : (n,) float32 target (SOH)
    groups     : (n,) object battery_id strings (for held_bids in output)
    cidx       : (n,) int64 cycle indices
    splitter   : sklearn GroupSplit-compatible splitter
    cv_groups  : optional array passed to splitter.split() — use ds_groups for LODO
    n_estimators, max_depth, min_samples_leaf, max_features, random_state
                 RF hyperparameters matching the nb03 practical RF defaults

    Returns
    -------
    list of fold dicts (same schema as run_grouped_cv):
        held_bids, held_group, val_bid (None),
        y_te, y_pred, cidx_te,
        train_curve (None), val_curve (None), best_epoch (None),
        metrics  (mae, rmse, r2, skill, spearman, naive_mae, n)
    """
    from sklearn.ensemble import RandomForestRegressor

    split_groups = cv_groups if cv_groups is not None else groups
    n_folds = splitter.get_n_splits(feat, groups=split_groups)
    folds: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(feat, groups=split_groups)
    ):
        feat_tr, y_tr, cidx_tr = feat[train_idx], y[train_idx], cidx[train_idx]
        feat_te, y_te, cidx_te = feat[test_idx], y[test_idx], cidx[test_idx]

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(feat_tr, y_tr)
        y_pred = rf.predict(feat_te).astype(np.float32)

        metrics = lobo_metrics(y_te, y_pred, cidx_te, y_tr.mean())

        held_bids = sorted(set(groups[test_idx]))
        held_group = sorted(set(split_groups[test_idx]))
        held_str = ",".join(held_group[:3]) + ("…" if len(held_group) > 3 else "")

        if progress:
            print(
                f"  [{fold_idx + 1:2d}/{n_folds}] held={held_str:20s}  "
                f"MAE={metrics['mae']:.4f}  R²={metrics['r2']:.3f}"
            )

        folds.append({
            "held_bids": held_bids,
            "held_group": held_group,
            "val_bid": None,
            "y_te": y_te,
            "y_pred": y_pred,
            "cidx_te": cidx_te,
            "train_curve": None,
            "val_curve": None,
            "best_epoch": None,
            "metrics": metrics,
        })

    return folds


# ---------------------------------------------------------------------------
# Extended plotting helpers (work with held_bids / held_group fold keys)
# ---------------------------------------------------------------------------


def per_fold_scatter_ext(
    folds: list[dict],
    title: str,
    save_path: Path | None = None,
) -> None:
    """Per-fold scatter (predicted vs actual SOH) for extended multi-dataset folds.

    Colours by dataset group: nasa → steelblue, calce / other → darkorange.
    Subplot title shows held battery IDs and per-fold MAE / R².
    Diagonal y=x reference line.

    Works with fold dicts from both run_grouped_cv and run_rf_grouped_cv.
    """
    import matplotlib.pyplot as plt

    DS_COLORS = {"nasa": "steelblue"}  # everything else → darkorange

    n = len(folds)
    nc = 4
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(
        nr, nc, figsize=(4 * nc, 3.5 * nr), constrained_layout=True
    )
    axes_flat = np.array(axes).flatten() if nr > 1 or nc > 1 else np.array([axes])

    for ax, fold in zip(axes_flat, folds):
        held_g = fold.get("held_group") or []
        color = DS_COLORS.get(held_g[0] if held_g else "", "darkorange")

        ax.scatter(fold["y_te"], fold["y_pred"], s=8, alpha=0.5, color=color)
        lo = float(min(fold["y_te"].min(), fold["y_pred"].min())) - 0.02
        hi = float(max(fold["y_te"].max(), fold["y_pred"].max())) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)

        m = fold["metrics"]
        bids = fold.get("held_bids") or held_g
        bid_str = ",".join(str(b) for b in bids[:3])
        if len(bids) > 3:
            bid_str += "…"
        ax.set_title(f"{bid_str}\nMAE={m['mae']:.4f}  R²={m['r2']:.3f}", fontsize=7)
        ax.set_xlabel("True SOH", fontsize=7)
        ax.set_ylabel("Pred SOH", fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[len(folds) :]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=11)
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def per_fold_loss_curves_ext(
    folds: list[dict],
    title: str,
    save_path: Path | None = None,
) -> None:
    """Per-fold train/val L1 loss curves for extended DL folds.

    Silently skips RF folds (train_curve=None).  Raises only if ALL folds are RF.
    """
    import matplotlib.pyplot as plt

    dl_folds = [f for f in folds if f.get("train_curve") is not None]
    if not dl_folds:
        print("  per_fold_loss_curves_ext: no DL folds found — skipped.")
        return

    n = len(dl_folds)
    nc = 4
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3 * nr), constrained_layout=True)
    axes_flat = np.array(axes).flatten() if nr > 1 or nc > 1 else np.array([axes])

    for ax, fold in zip(axes_flat, dl_folds):
        tr = fold["train_curve"]
        vl = fold["val_curve"]
        ep = fold["best_epoch"]
        epochs = range(1, len(tr) + 1)
        ax.plot(epochs, tr, label="train", lw=1.2)
        ax.plot(epochs, vl, label="val", lw=1.2)
        if ep and ep > 0:
            ax.axvline(ep, color="gray", lw=0.8, ls="--")

        bids = fold.get("held_bids") or fold.get("held_group") or []
        bid_str = ",".join(str(b) for b in bids[:2])
        if len(bids) > 2:
            bid_str += "…"
        ax.set_title(f"{bid_str}  (best ep={ep})", fontsize=8)
        ax.set_xlabel("Epoch", fontsize=7)
        ax.set_ylabel("L1 loss", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)

    for ax in axes_flat[len(dl_folds) :]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=11)
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
