"""
Generate summary comparison charts for the presentation narrative.
No model training required — uses hardcoded numbers from nb04 + nb05 outputs.

Run: uv run python scripts/gen_comparison_charts.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("results/extended_tier1")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── 1. GKF-6 vs LODO model comparison ─────────────────────────────────────────
# Numbers from nb04 (authoritative multi-model run).
# GKF-6 std from nb04 comparison table.
models = ["RF", "BiGRU", "CNN-LSTM", "Transformer", "Attn-GRU"]
gkf_mae  = [0.0735, 0.0381, 0.0384, 0.0332, 0.0367]
gkf_std  = [0.0558, 0.0153, 0.0179, 0.0259, 0.0145]
lodo_mae = [0.1469, 0.0412, 0.0711, 0.0826, 0.0661]
# LODO std is across 2 folds only; use asymmetric bars from the two held-out values
lodo_lo  = [min(0.1414, 0.1524), min(0.0397, 0.0426), min(0.0849, 0.0573),
            min(0.0605, 0.1046), min(0.0608, 0.0713)]
lodo_hi  = [max(0.1414, 0.1524), max(0.0397, 0.0426), max(0.0849, 0.0573),
            max(0.0605, 0.1046), max(0.0608, 0.0713)]

x = np.arange(len(models))
w = 0.35

# Colours: blue for in-dist, orange for transfer
c_gkf  = "#4C72B0"
c_lodo = "#DD8452"

fig, ax = plt.subplots(figsize=(9, 5))

bars1 = ax.bar(x - w/2, gkf_mae, w, label="GroupKFold-6 (in-distribution)",
               color=c_gkf, alpha=0.88, zorder=3)
ax.errorbar(x - w/2, gkf_mae, yerr=gkf_std, fmt="none", color="k",
            capsize=4, linewidth=1.2, zorder=4)

lodo_err_lo = [m - lo for m, lo in zip(lodo_mae, lodo_lo)]
lodo_err_hi = [hi - m  for m, hi in zip(lodo_mae, lodo_hi)]
bars2 = ax.bar(x + w/2, lodo_mae, w, label="LODO (cross-dataset transfer)",
               color=c_lodo, alpha=0.88, zorder=3)
ax.errorbar(x + w/2, lodo_mae, yerr=[lodo_err_lo, lodo_err_hi], fmt="none",
            color="k", capsize=4, linewidth=1.2, zorder=4)

# Annotate transfer gap for Transformer
gap_idx = 3  # Transformer
ax.annotate(
    "Best in-dist\n→ collapses\non transfer",
    xy=(x[gap_idx] + w/2, lodo_mae[gap_idx]),
    xytext=(x[gap_idx] + 0.55, lodo_mae[gap_idx] + 0.012),
    fontsize=9, color="#AA3333",
    arrowprops=dict(arrowstyle="->", color="#AA3333", lw=1.2),
)

# Annotate BiGRU as the balance point
ax.annotate(
    "Most robust\ntransfer",
    xy=(x[1] + w/2, lodo_mae[1]),
    xytext=(x[1] + 0.55, lodo_mae[1] + 0.018),
    fontsize=9, color="#2A6E2A",
    arrowprops=dict(arrowstyle="->", color="#2A6E2A", lw=1.2),
)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel("MAE (lower is better)", fontsize=12)
ax.set_title("Model comparison: within-pool accuracy vs cross-dataset transfer\n"
             "(22 cells, NASA + CALCE, GroupKFold-6 / LODO)", fontsize=12)
ax.legend(fontsize=10, loc="upper right")
ax.set_ylim(0, 0.20)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
out_path = OUT / "comparison_gkf_vs_lodo.png"
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()


# ── 2. LODO per-model breakdown: hold-CALCE vs hold-NASA ──────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))

hold_calce = [0.1414, 0.0397, 0.0849, 0.0605, 0.0608]
hold_nasa  = [0.1524, 0.0426, 0.0573, 0.1046, 0.0713]

bars_c = ax.bar(x - w/2, hold_calce, w, label="Hold-out: CALCE (train on NASA)",
                color="#5BA85A", alpha=0.88, zorder=3)
bars_n = ax.bar(x + w/2, hold_nasa,  w, label="Hold-out: NASA (train on CALCE)",
                color="#C44E52", alpha=0.88, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel("MAE (lower is better)", fontsize=12)
ax.set_title("Cross-dataset transfer: CALCE ↔ NASA\n"
             "(BiGRU is the only model that survives both directions)", fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0, 0.22)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
out_path = OUT / "lodo_calce_vs_nasa.png"
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()


# ── 3. Augmentation progression ───────────────────────────────────────────────
# Numbers from nb04 (plain BiGRU) and nb05 (aug sweep).
aug_labels = [
    "Plain BiGRU\n(no aug)",
    "BiGRU +\nrate_warp",
    "BiGRU +\nOptuna aug",
    "Optuna aug\n+ 1-shot (k=1)",
]
aug_lodo_calce = [0.0397, 0.0587, 0.0514, 0.0681]
aug_lodo_nasa  = [0.0426, 0.0703, 0.0646, 0.0452]

xa = np.arange(len(aug_labels))

fig, ax = plt.subplots(figsize=(9, 4.5))

bc = ax.bar(xa - w/2, aug_lodo_calce, w, label="Hold-out: CALCE",
            color="#5BA85A", alpha=0.88, zorder=3)
bn = ax.bar(xa + w/2, aug_lodo_nasa,  w, label="Hold-out: NASA",
            color="#C44E52", alpha=0.88, zorder=3)

# Highlight the recommended operating point
rect = plt.Rectangle((xa[2] - 0.5, 0), 1.0, 0.12, linewidth=2,
                      edgecolor="#4C72B0", facecolor="#4C72B0", alpha=0.08, zorder=2)
ax.add_patch(rect)
ax.text(xa[2], 0.102, "Recommended\nzero-shot", ha="center", va="bottom",
        color="#4C72B0", fontsize=9, fontweight="bold")

ax.set_xticks(xa)
ax.set_xticklabels(aug_labels, fontsize=10)
ax.set_ylabel("LODO MAE (lower is better)", fontsize=12)
ax.set_title("Effect of augmentation on cross-dataset transfer\n"
             "(rate_warp teaches C-rate invariance; time_warp teaches duration invariance)",
             fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0, 0.115)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
out_path = OUT / "augmentation_progression.png"
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()


# ── 4. Phase 1 baseline table — bar chart of NASA practical phase models ───────
# From nb03: practical phase, rul_frac LOBO
p1_models = ["Ridge", "Random\nForest", "GBM",
             "Macro-clock\nTCN", "VG-LSTM\n(NASA only)"]
p1_mae = [0.136, 0.032, 0.035, 0.042, 0.062]
p1_r2  = [0.560, 0.966, 0.961, 0.935, 0.578]
colors = ["#AAAAAA", "#4C72B0", "#AAAAAA", "#AAAAAA", "#DD8452"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

bars = ax1.bar(p1_models, p1_mae, color=colors, zorder=3, alpha=0.88)
ax1.set_ylabel("LOBO MAE (lower is better)", fontsize=11)
ax1.set_title("NASA phase: model comparison", fontsize=11)
ax1.set_ylim(0, 0.18)
ax1.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax1.set_axisbelow(True)
ax1.tick_params(axis="x", labelsize=9)
ax1.annotate("RF wins\non 13 cells", xy=(1, 0.032), xytext=(1, 0.09),
             ha="center", color="#4C72B0", fontsize=9, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#4C72B0"))

bars2 = ax2.bar(p1_models, p1_r2, color=colors, zorder=3, alpha=0.88)
ax2.set_ylabel("LOBO R² (higher is better)", fontsize=11)
ax2.set_title("NASA phase: R² comparison", fontsize=11)
ax2.set_ylim(0, 1.05)
ax2.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax2.set_axisbelow(True)
ax2.tick_params(axis="x", labelsize=9)

plt.tight_layout()
out_path = OUT / "nasa_phase_baseline_comparison.png"
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()

print("\nAll charts generated.")
