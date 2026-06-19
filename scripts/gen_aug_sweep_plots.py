"""
Clean augmentation sweep: run all 4 configs on LODO from scratch, then
generate the augmentation progression bar chart + BiGRU scatter/trajectory plots.

Configs (all identical except augmentation):
  A. plain    — no augmentation
  B. rw_only  — rate_warp ±25%, jitter σ=0.01  (nb04 anchor)
  C. optuna   — rate_warp ±7%, jitter σ=0.006, time_warp ±11%  (nb05 Optuna best)
  D. one_shot — config C + 1-shot FC fine-tune on k=1 target cell

LODO: 2 folds (hold-CALCE, hold-NASA). Same inner-val logic for all configs:
  last-alphabetical battery from the training distribution (never from held dataset).

Runtime: ~10 min per fold on GPU → ~40-80 min total (8 LODO folds across 4 configs).
For one_shot, fine-tuning adds ~2 min per fold.

Run: uv run python scripts/gen_aug_sweep_plots.py
"""

import copy
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from src.vg_extended import load_npz_ext, run_grouped_cv, per_fold_scatter_ext
from src.vg_models import VGGRUReg, predict_vg
from src.vg_augment import make_augment
from src.voltage_grid import lobo_metrics

# ── config ─────────────────────────────────────────────────────────────────────
NPZ_PATH = ROOT / "data/processed/vg_extended_tier1.npz"
OUT      = ROOT / "results/extended_tier1"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")

HIDDEN   = 47
DROPOUT  = 0.35
EPOCHS   = 300
PATIENCE = 50
SEED     = 42

# 1-shot: number of cells taken from held-out dataset for FC fine-tune
K_SHOT   = 1
# 1-shot fine-tune epochs / patience
FT_EPOCHS   = 100
FT_PATIENCE = 20
FT_LR       = 3e-4

# ── augmentation definitions ───────────────────────────────────────────────────
AUGS = {
    "plain": make_augment(
        rate_warp_lo=1.0, rate_warp_hi=1.0,
        jitter_sigma=0.0,
        time_warp_lo=1.0, time_warp_hi=1.0,
    ),
    "rw_only": make_augment(
        rate_warp_lo=0.75, rate_warp_hi=1.25,
        jitter_sigma=0.01,
        time_warp_lo=1.0, time_warp_hi=1.0,
    ),
    "optuna": make_augment(
        rate_warp_lo=1 - 0.07, rate_warp_hi=1 + 0.07,
        jitter_sigma=0.006,
        time_warp_lo=1 - 0.11, time_warp_hi=1 + 0.11,
    ),
}

# ── load data ──────────────────────────────────────────────────────────────────
print(f"Device: {DEVICE}")
print(f"Loading {NPZ_PATH} …")
X, mask, y, groups, cidx, ds_groups = load_npz_ext(NPZ_PATH)
print(f"X: {X.shape}  y: [{y.min():.3f}, {y.max():.3f}]  "
      f"cells: {len(set(groups))}  datasets: {set(ds_groups)}\n")


# ── helper: fine-tune FC head on k cells from the held-out dataset ─────────────
def one_shot_finetune(
    base_model: nn.Module,
    X_held: np.ndarray,
    mask_held: np.ndarray,
    y_held: np.ndarray,
    groups_held: np.ndarray,
    scaler: StandardScaler,
    device: torch.device,
    k: int = 1,
) -> tuple[nn.Module, np.ndarray]:
    """Fine-tune only the FC head on k cells from the held-out set.

    Returns
    -------
    ft_model : fine-tuned copy of base_model
    eval_mask : bool mask over X_held/y_held selecting the EVALUATION cells
                (held_out minus the k fine-tune cells)
    """
    unique_bids = sorted(set(groups_held))
    # Pick last-alphabetical k cells as fine-tune cells (deterministic)
    ft_bids  = unique_bids[-k:]
    eval_bids = unique_bids[:-k]

    ft_mask   = np.isin(groups_held, ft_bids)
    eval_mask = np.isin(groups_held, eval_bids)

    if eval_mask.sum() == 0:
        # Edge case: held-out dataset has only k cells — evaluate on ft cells too
        eval_mask = np.ones(len(groups_held), dtype=bool)

    X_ft   = X_held[ft_mask]
    m_ft   = mask_held[ft_mask]
    y_ft   = y_held[ft_mask]

    # Scale using the same scaler (already fit on train; apply to ft samples)
    n, L, C = X_ft.shape
    X_ft_2d = X_ft.reshape(-1, C)
    X_ft_sc = scaler.transform(X_ft_2d).reshape(n, L, C).astype(np.float32)
    # Re-zero pad positions
    X_ft_sc[~m_ft] = 0.0

    # Freeze everything except fc
    ft_model = copy.deepcopy(base_model)
    for name, param in ft_model.named_parameters():
        param.requires_grad = (name.startswith("fc."))

    ft_model.to(device).train()
    opt = torch.optim.Adam(ft_model.fc.parameters(), lr=FT_LR)
    criterion = nn.L1Loss()

    Xt = torch.tensor(X_ft_sc, dtype=torch.float32)
    mt = torch.tensor(m_ft,    dtype=torch.bool)
    yt = torch.tensor(y_ft,    dtype=torch.float32).unsqueeze(1)

    best_loss = float("inf")
    best_state = copy.deepcopy(ft_model.state_dict())
    no_improve = 0

    for ep in range(FT_EPOCHS):
        ft_model.train()
        opt.zero_grad()
        pred = ft_model(Xt.to(device), mt.to(device))
        loss = criterion(pred, yt.to(device))
        loss.backward()
        opt.step()

        ft_model.eval()
        with torch.no_grad():
            val_loss = criterion(ft_model(Xt.to(device), mt.to(device)), yt.to(device)).item()
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(ft_model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= FT_PATIENCE:
            break

    ft_model.load_state_dict(best_state)
    return ft_model, eval_mask


# ── helper: run LODO for one aug config and return per-fold dicts ──────────────
def run_lodo(aug_name: str) -> list[dict]:
    aug_fn = AUGS[aug_name]
    print(f"\n{'='*60}")
    print(f"  Config: {aug_name}")
    print(f"{'='*60}")

    folds = run_grouped_cv(
        lambda: VGGRUReg(n_features=3, hidden=HIDDEN, dropout=DROPOUT),
        X, mask, y, groups, cidx,
        DEVICE,
        LeaveOneGroupOut(),
        cv_groups=ds_groups,
        epochs=EPOCHS,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        patience=PATIENCE,
        scheduler="cosine",
        progress=True,
        verbose=True,
        seed=SEED,
        train_kwargs={"augment_fn": aug_fn},
    )

    print(f"\n  {aug_name} LODO summary:")
    for fold in folds:
        m = fold["metrics"]
        held = (fold.get("held_group") or ["?"])[0]
        print(f"    held={held:8s}  MAE={m['mae']:.4f}  R²={m['r2']:.3f}")
    return folds


# ── helper: run 1-shot on top of a trained-fold list ─────────────────────────
def run_one_shot(base_folds: list[dict]) -> list[dict]:
    """For each LODO fold, fine-tune the FC head on K_SHOT target cells."""
    print(f"\n{'='*60}")
    print(f"  Config: one_shot  (k={K_SHOT} fine-tune cells per fold)")
    print(f"{'='*60}")

    # We need to rebuild the scaler from training data for each fold.
    # run_grouped_cv doesn't return the scaler, so we refit it here.
    # The scaler was fit on (masked) training rows — replicate that logic.
    splitter = LeaveOneGroupOut()
    ft_folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(X, groups=ds_groups)
    ):
        fold = base_folds[fold_idx]
        held_ds = (fold.get("held_group") or ["?"])[0]

        # Refit scaler on training data (same logic as run_grouped_cv)
        X_tr   = X[train_idx]
        m_tr   = mask[train_idx]
        y_tr   = y[train_idx]
        n, L, C = X_tr.shape
        X_2d   = X_tr.reshape(-1, C)
        valid  = m_tr.reshape(-1)
        scaler = StandardScaler()
        scaler.fit(X_2d[valid])

        # Scale training data (needed to reconstruct the trained model's input space)
        # We don't actually retrain — we use the already-trained fold model.
        # Just refit scaler to match what the model was trained on.

        # Held data (unscaled)
        X_held_raw  = X[test_idx]
        m_held      = mask[test_idx]
        y_held      = y[test_idx]
        g_held      = groups[test_idx]
        ci_held     = cidx[test_idx]

        # Scale held data with the same scaler
        n_h, L_h, C_h = X_held_raw.shape
        X_held_2d   = X_held_raw.reshape(-1, C_h)
        X_held_sc   = scaler.transform(X_held_2d).reshape(n_h, L_h, C_h).astype(np.float32)
        X_held_sc[~m_held] = 0.0

        # We need the trained base model from the fold.
        # run_grouped_cv doesn't return the model object, only predictions.
        # So we load fold predictions as the "no fine-tune" baseline and
        # re-train a fresh model here only for the fine-tuned version.
        # For a clean apples-to-apples comparison we retrain the base model
        # for this fold and then fine-tune it.

        print(f"\n  Fold {fold_idx}: held={held_ds}")
        aug_fn = AUGS["optuna"]

        # Rebuild inner-val logic (same as run_grouped_cv)
        avail_bids = sorted(set(groups[train_idx]))
        bid_to_ds  = {b: ds_groups[train_idx][groups[train_idx] == b][0]
                      for b in avail_bids}
        ds_counts  = {}
        for b, d in bid_to_ds.items():
            ds_counts[d] = ds_counts.get(d, 0) + (groups[train_idx] == b).sum()
        majority_ds = max(ds_counts, key=ds_counts.get)
        majority_bids = sorted([b for b, d in bid_to_ds.items() if d == majority_ds])
        val_bid = majority_bids[-1]

        val_mask  = groups[train_idx] == val_bid
        tr_mask   = ~val_mask
        X_tr_net  = X_tr[tr_mask];  m_tr_net = m_tr[tr_mask];  y_tr_net = y_tr[tr_mask]
        X_val_net = X_tr[val_mask]; m_val_net = m_tr[val_mask]; y_val_net = y_tr[val_mask]

        # Scale
        n2, L2, C2 = X_tr_net.shape
        X2d = X_tr_net.reshape(-1, C2)
        scaler2 = StandardScaler()
        scaler2.fit(X2d[m_tr_net.reshape(-1)])
        def sc(Xa, ma):
            n_, L_, C_ = Xa.shape
            out = scaler2.transform(Xa.reshape(-1, C_)).reshape(n_, L_, C_).astype(np.float32)
            out[~ma] = 0.0
            return out
        X_tr_sc  = sc(X_tr_net,  m_tr_net)
        X_val_sc = sc(X_val_net, m_val_net)
        X_held_sc2 = sc(X_held_raw, m_held)

        from src.sequence import train_model as _train
        fold_seed = fold_idx * 31 + SEED
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)

        base_model = VGGRUReg(n_features=3, hidden=HIDDEN, dropout=DROPOUT).to(DEVICE)
        _, _, _ = _train(
            base_model, X_tr_sc, m_tr_net, y_tr_net,
            X_val_sc, m_val_net, y_val_net,
            DEVICE,
            epochs=EPOCHS, batch_size=64, lr=1e-3, weight_decay=1e-4,
            patience=PATIENCE, scheduler="cosine", progress=True,
            augment_fn=aug_fn,
        )

        # Baseline (no fine-tune) predictions on all held cells
        base_preds = predict_vg(base_model, X_held_sc2, m_held, DEVICE)
        base_mae   = float(np.mean(np.abs(base_preds - y_held)))

        # Fine-tune on K_SHOT cells
        ft_model, eval_mask = one_shot_finetune(
            base_model, X_held_raw, m_held, y_held, g_held,
            scaler2, DEVICE, k=K_SHOT,
        )

        # Evaluate fine-tuned model on eval cells
        X_eval_sc = sc(X_held_raw[eval_mask], m_held[eval_mask])
        ft_preds  = predict_vg(ft_model, X_eval_sc, m_held[eval_mask], DEVICE)
        y_eval    = y_held[eval_mask]
        ft_mae    = float(np.mean(np.abs(ft_preds - y_eval)))

        from sklearn.metrics import r2_score
        ft_r2 = float(r2_score(y_eval, ft_preds))

        print(f"    val_bid={val_bid}  base_mae={base_mae:.4f}  "
              f"ft_mae={ft_mae:.4f}  ft_r2={ft_r2:.3f}  "
              f"n_eval={eval_mask.sum()}")

        ft_folds.append({
            "held_group": [held_ds],
            "held_bids":  g_held[eval_mask].tolist(),
            "val_bid":    val_bid,
            "y_te":       y_eval,
            "y_pred":     ft_preds,
            "cidx_te":    ci_held[eval_mask],
            "metrics":    {
                "mae":      ft_mae,
                "r2":       ft_r2,
                "n":        int(eval_mask.sum()),
                "skill":    float(1 - ft_mae / (np.mean(np.abs(y_eval - y_eval.mean())) + 1e-9)),
                "spearman": 0.0,  # skip for speed
                "rmse":     float(np.sqrt(np.mean((ft_preds - y_eval)**2))),
                "naive_mae": float(np.mean(np.abs(y_eval - y_eval.mean()))),
            },
        })

    print("\n  one_shot LODO summary:")
    for fold in ft_folds:
        m = fold["metrics"]
        held = (fold.get("held_group") or ["?"])[0]
        print(f"    held={held:8s}  MAE={m['mae']:.4f}  R²={m['r2']:.3f}")
    return ft_folds


# ── run all configs ────────────────────────────────────────────────────────────
results: dict[str, list[dict]] = {}
for cfg in ["plain", "rw_only", "optuna"]:
    results[cfg] = run_lodo(cfg)

results["one_shot"] = run_one_shot(results["optuna"])


# ── extract LODO MAE per fold per config ──────────────────────────────────────
def fold_mae(folds: list[dict], held_ds: str) -> float:
    for f in folds:
        if (f.get("held_group") or [""])[0] == held_ds:
            return f["metrics"]["mae"]
    return float("nan")

cfg_labels = ["Plain\nBiGRU", "BiGRU +\nrate_warp", "BiGRU +\nOptuna aug", "Optuna aug\n+ 1-shot (k=1)"]
cfg_keys   = ["plain", "rw_only", "optuna", "one_shot"]

calce_maes = [fold_mae(results[k], "calce") for k in cfg_keys]
nasa_maes  = [fold_mae(results[k], "nasa")  for k in cfg_keys]

print("\n\n=== FINAL RESULTS ===")
print(f"{'Config':<22}  {'hold-CALCE':>12}  {'hold-NASA':>10}  {'LODO mean':>10}")
for lbl, kk, cm, nm in zip(cfg_labels, cfg_keys, calce_maes, nasa_maes):
    lbl_flat = lbl.replace("\n", " ")
    mean = (cm + nm) / 2
    print(f"  {lbl_flat:<20}  {cm:>12.4f}  {nm:>10.4f}  {mean:>10.4f}")


# ── augmentation progression bar chart ────────────────────────────────────────
xa = np.arange(len(cfg_labels))
w  = 0.35

plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(figsize=(9, 4.5))

ax.bar(xa - w/2, calce_maes, w, label="Hold-out: CALCE (train on NASA)",
       color="#5BA85A", alpha=0.88, zorder=3)
ax.bar(xa + w/2, nasa_maes,  w, label="Hold-out: NASA  (train on CALCE)",
       color="#C44E52", alpha=0.88, zorder=3)

# Annotate bar tops
for i, (cm, nm) in enumerate(zip(calce_maes, nasa_maes)):
    ax.text(xa[i] - w/2, cm + 0.003, f"{cm:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(xa[i] + w/2, nm + 0.003, f"{nm:.3f}", ha="center", va="bottom", fontsize=8)

# Highlight recommended zero-shot operating point
rect = plt.Rectangle((xa[2] - 0.5, 0), 1.0, max(calce_maes + nasa_maes) * 1.02,
                      linewidth=2, edgecolor="#4C72B0", facecolor="#4C72B0",
                      alpha=0.07, zorder=2)
ax.add_patch(rect)
ax.text(xa[2], max(calce_maes[2], nasa_maes[2]) + 0.012,
        "Recommended\nzero-shot", ha="center", va="bottom",
        color="#4C72B0", fontsize=9, fontweight="bold")

ax.set_xticks(xa)
ax.set_xticklabels(cfg_labels, fontsize=10)
ax.set_ylabel("LODO MAE (lower is better)", fontsize=12)
ax.set_title("Effect of augmentation on cross-dataset transfer (BiGRU, LODO)\n"
             "rate_warp = C-rate invariance  |  time_warp = duration invariance",
             fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, max(calce_maes + nasa_maes) * 1.35)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
out_path = OUT / "augmentation_progression.png"
fig.savefig(out_path, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.close()


# ── scatter plots for the optuna config (best zero-shot) ──────────────────────
per_fold_scatter_ext(
    results["optuna"],
    title="BiGRU + Optuna aug — LODO per-fold scatter (SOH)",
    save_path=OUT / "bigru_lodo_per_fold.png",
)


# ── SOH trajectory plots for optuna config ────────────────────────────────────
for fold in results["optuna"]:
    held_ds   = (fold.get("held_group") or ["?"])[0]
    y_te      = fold["y_te"]
    y_pred    = fold["y_pred"]
    cidx_te   = fold["cidx_te"]

    held_mask_global = ds_groups == held_ds
    held_g_global    = groups[held_mask_global]
    held_ci_global   = cidx[held_mask_global]
    assert len(held_g_global) == len(y_te)

    unique_bids = sorted(set(held_g_global))
    ncols = min(4, len(unique_bids))
    nrows = (len(unique_bids) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows),
                              constrained_layout=True)
    axes_flat = np.array(axes).flatten() if (nrows > 1 or ncols > 1) else np.array([axes])

    for ax, bid in zip(axes_flat, unique_bids):
        sel   = held_g_global == bid
        ci    = held_ci_global[sel]
        yt    = y_te[sel]
        yp    = y_pred[sel]
        order = np.argsort(ci)
        ci, yt, yp = ci[order], yt[order], yp[order]
        mae_b = float(np.mean(np.abs(yt - yp)))

        ax.plot(ci, yt,  "k-",  lw=1.8, label="True SOH")
        ax.plot(ci, yp, "--",   lw=1.5, color="#DD8452", label="Predicted")
        ax.axhline(0.8, color="red", lw=0.8, linestyle=":", alpha=0.7, label="EOL (80%)")
        ax.set_title(f"{bid}  MAE={mae_b:.3f}", fontsize=9)
        ax.set_xlabel("Cycle index", fontsize=8)
        ax.set_ylabel("SOH", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right")
        lo = max(0.0, float(min(yt.min(), yp.min())) - 0.05)
        hi = float(max(yt.max(), yp.max())) + 0.05
        ax.set_ylim(lo, hi)

    for ax in axes_flat[len(unique_bids):]:
        ax.set_visible(False)

    src_ds = "NASA" if held_ds == "calce" else "CALCE"
    fig.suptitle(
        f"BiGRU + Optuna aug — SOH trajectories  "
        f"(hold-{held_ds.upper()}, trained on {src_ds})",
        fontsize=11,
    )
    traj_path = OUT / f"bigru_lodo_trajectory_{held_ds}.png"
    fig.savefig(traj_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {traj_path}")
    plt.close(fig)


print("\nAll done. Files in", OUT)
