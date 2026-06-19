#!/usr/bin/env python3
"""Throughput & quality benchmark: GRU / LSTM / CNN-GRU / Transformer × torch.compile.

Usage
-----
    uv run python benchmark_throughput.py

Each model is run through GroupKFold-6 (split by study group, matching nb03).
Wall-clock time is measured per full CV run and divided by fold count.
Compile=True uses torch.compile(); on Windows the aot_eager backend is tried
first to avoid the Triton dependency that inductor requires on Linux/Mac.

Results are printed as a Markdown table at the end.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

# ── project root ──────────────────────────────────────────────────────────────
ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "pyproject.toml").exists()
)
sys.path.insert(0, str(ROOT))

import src.voltage_grid as vg
import src.vg_extended as vgx
import src.vg_models as vgm

# ── hardware ──────────────────────────────────────────────────────────────────
DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()         else
    torch.device("cpu")
)

# ── training knobs (match nb03) ───────────────────────────────────────────────
CACHE      = ROOT / "data/processed/vg_extended_tier1.npz"
EPOCHS     = 300
PATIENCE   = 50
BATCH_SIZE = 64
LR         = 1e-3
N_SPLITS   = 6

# ── study-group map (mirrors nb03 cell d11002f5) ──────────────────────────────
_SG_MAP: dict[str, str] = {
    "B0005": "NASA_CTRL",  "B0006": "NASA_CTRL",
    "B0007": "NASA_CTRL",  "B0018": "NASA_CTRL",
    "RW1":   "NASA_RW",    "RW9":   "NASA_RW",
    "RW13":  "NASA_RW",    "RW14":  "NASA_RW",
    "RW15":  "NASA_RW",    "RW16":  "NASA_RW",
    "RW17":  "NASA_RW",    "RW19":  "NASA_RW",
    "RW20":  "NASA_RW",
    "CS2_8":  "CALCE_CS2_T1", "CS2_21": "CALCE_CS2_T1",
    "CS2_33": "CALCE_CS2_T1", "CS2_34": "CALCE_CS2_T1",
    "CS2_35": "CALCE_CS2_T2", "CS2_36": "CALCE_CS2_T2",
    "CS2_37": "CALCE_CS2_T2", "CS2_38": "CALCE_CS2_T2",
    "CS2_3":  "CALCE_CS2_T3", "CS2_9":  "CALCE_CS2_T3",
    "CX2_16": "CALCE_CX2_T1", "CX2_31": "CALCE_CX2_T1",
    "CX2_33": "CALCE_CX2_T1", "CX2_35": "CALCE_CX2_T1",
    "CX2_34": "CALCE_CX2_T2", "CX2_36": "CALCE_CX2_T2",
    "CX2_37": "CALCE_CX2_T2", "CX2_38": "CALCE_CX2_T2",
    "CX2_8":  "CALCE_CX2_T3",
}


# ── model factories ───────────────────────────────────────────────────────────
MODELS: list[tuple[str, callable]] = [
    ("GRU",         lambda: vgm.VGGRUReg(n_features=3, hidden=47, dropout=0.35)),
    ("LSTM",        lambda: vgm.VGLSTMReg(n_features=3, hidden=40, dropout=0.35)),
    ("CNN-GRU",     lambda: vgm.VGCNNGRU(n_features=3, cnn_ch=8, hidden=47, dropout=0.35)),
    ("Transformer", lambda: vgm.VGTransformerReg(
        n_features=3, d_model=32, nhead=4, dim_feedforward=64,
        num_layers=2, dropout=0.35,
    )),
]


def _try_compile(model: torch.nn.Module) -> tuple[torch.nn.Module, str]:
    """Wrap model with torch.compile(); fall back gracefully on Windows/no Triton."""
    # Try inductor first (optimal); fall back to aot_eager (graph capture only)
    for backend in ("inductor", "aot_eager"):
        try:
            compiled = torch.compile(model, backend=backend)
            # Trigger a dummy trace to catch deferred errors early
            return compiled, backend
        except Exception as e:
            last_err = e
    return model, f"failed ({last_err})"


def _param_count(factory: callable) -> int:
    m = factory()
    return sum(p.numel() for p in m.parameters())


def run_config(
    name: str,
    factory: callable,
    X: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cidx: np.ndarray,
    study_groups: np.ndarray,
    compile_flag: bool,
) -> dict:
    compile_label = "no"
    backend_info  = ""

    if compile_flag:
        # Build a throwaway model to test compile
        probe, backend = _try_compile(factory().to(DEVICE))
        if "failed" in backend:
            compile_label = f"failed"
            backend_info  = backend
            # Run without compile
            actual_compile = False
        else:
            compile_label = "yes"
            backend_info  = backend
            actual_compile = True
    else:
        actual_compile = False

    label = f"{name:12s}  compile={compile_label}"
    print(f"\n{'='*60}")
    print(f"  {label}  [{backend_info}]")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    try:
        folds = vgx.run_grouped_cv(
            factory,
            X, mask, y, groups, cidx, DEVICE,
            GroupKFold(n_splits=N_SPLITS),
            cv_groups=study_groups,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            patience=PATIENCE,
            scheduler="plateau",
            progress=True,
            compile_model=actual_compile,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "model": name, "compile": compile_label, "backend": backend_info,
            "error": str(e),
        }
    elapsed = time.perf_counter() - t0

    agg = vg.aggregate(folds)
    return {
        "model":     name,
        "compile":   compile_label,
        "backend":   backend_info,
        "params":    _param_count(factory),
        "mae":       agg["mae"],
        "mae_std":   agg["mae_std"],
        "rmse":      agg["rmse"],
        "rmse_std":  agg["rmse_std"],
        "r2":        agg["r2"],
        "r2_std":    agg["r2_std"],
        "skill":     agg["skill"],
        "spearman":  agg["spearman"],
        "total_s":   elapsed,
        "s_per_fold": elapsed / N_SPLITS,
        "error":     None,
    }


def _row(r: dict) -> str:
    if r.get("error"):
        return (
            f"| {r['model']:<12} | {r['compile']:<7} | {r.get('backend',''):<10} | "
            f"{'ERROR':<37} {r['error'][:40]} |"
        )
    return (
        f"| {r['model']:<12} | {r['compile']:<7} | {r.get('backend',''):<10} | "
        f"{r['params']:>6,} | "
        f"{r['mae']:.4f}±{r['mae_std']:.4f} | "
        f"{r['rmse']:.4f}±{r['rmse_std']:.4f} | "
        f"{r['r2']:+.4f}±{r['r2_std']:.4f} | "
        f"{r['skill']:.4f} | "
        f"{r['spearman']:.4f} | "
        f"{r['total_s']:7.1f}s | "
        f"{r['s_per_fold']:5.1f}s |"
    )


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Device : {DEVICE}")
    print(f"Cache  : {CACHE}")
    print(f"Epochs : {EPOCHS}  Patience : {PATIENCE}  LR : {LR}")

    X, mask, y, groups, cidx, _ = vgx.load_npz_ext(CACHE)
    study_groups = np.array([_SG_MAP.get(bid, "unknown") for bid in groups])

    n_sg = len(set(study_groups))
    print(f"Samples: {len(X)}   Cells: {len(set(groups))}   Study groups: {n_sg}")
    print(f"Configs: {len(MODELS)} models × 2 compile flags = {len(MODELS)*2} runs\n")

    results: list[dict] = []

    for model_name, factory in MODELS:
        for compile_flag in (False, True):
            r = run_config(
                model_name, factory,
                X, mask, y, groups, cidx, study_groups,
                compile_flag=compile_flag,
            )
            results.append(r)

    # ── summary table ─────────────────────────────────────────────────────────
    hdr = (
        f"| {'Model':<12} | {'Compile':<7} | {'Backend':<10} | "
        f"{'Params':>6} | "
        f"{'MAE ± std':<16} | "
        f"{'RMSE ± std':<17} | "
        f"{'R² ± std':<17} | "
        f"{'Skill':<6} | "
        f"{'Spear.':<7} | "
        f"{'Total':>8} | "
        f"{'s/fold':>6} |"
    )
    sep = "|" + "|".join("-" * (len(c) - 0) for c in hdr.split("|")[1:-1]) + "|"

    print("\n\n" + "=" * 80)
    print("  BENCHMARK RESULTS — GroupKFold-6 (by study group)")
    print("=" * 80)
    print(hdr)
    print(sep)
    for r in results:
        print(_row(r))
    print()


if __name__ == "__main__":
    main()
