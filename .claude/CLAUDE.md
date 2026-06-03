# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

EFREI "Machine Learning for Data Engineers" course project (instructor: Stephany Rajeh, Ph.D.).

**Topic:** Battery Health Prediction — build a time-series model to predict the lifespan and efficiency of batteries used in renewable energy systems.

**Deliverables:**
- 15-minute presentation (out of scope for this repo)
- Code made public on GitHub (link must appear in the PPT)

## Model Progression

The instructor explicitly recommends this order — follow it:
1. ✅ Baseline: Ridge, Random Forest, XGBoost (see `notebooks/03_train_summary.md` for results)
2. 🚧 Advanced: PyTorch LSTM or Transformer for time-series prediction

Always report metrics (MAE, RMSE, R²) for every model and compare them in a table.

Validation: use Leave-One-Battery-Out (LOBO) CV. Never split train/test by time within a single battery — always hold out complete batteries.

## Stack

- Data: `pandas`, `numpy`, `scipy`
- Baselines: `scikit-learn`, `xgboost`, `optuna` (hyperparameter tuning)
- Deep learning: `PyTorch` (pending)
- Notebooks: Jupyter (exploration) → `.py` scripts (final reproducible pipeline)
- Package manager: `uv` — always use `uv run`, `uv add`; never `pip` or bare `python3`

## Project Structure

```
data/          raw and processed datasets (gitignore large files)
notebooks/     exploratory and analysis notebooks
src/           reusable Python modules (preprocessing, features, models)
src/battery.py core loader, feature engineering, RUL computation (362 lines)
models/        saved model checkpoints
results/       metrics, plots, evaluation outputs
```

## Datasets

NASA PCoE Li-ion battery datasets are already in `data/`. Dataset structure: `@dataset-structure.md`.

**Retained batteries (13):** Controlled B0005–B0007, B0018; Randomized RW1, RW9, RW13–RW17, RW19–RW20.

**Excluded:** Hot group (43°C, never reach EOL), cold group (4°C, inconsistent mixed load), RW2/RW18 (sensor errors), B0041 (anomalous starting capacity).

## Targets & Modeling Phases

Two targets:
- **`rul_frac`** (lifespan): remaining cycles / EOL index, normalised 0–1. EOL = 80% capacity retention.
- **`energy_retention`** (efficiency): cycle Wh / baseline Wh — captures voltage sag, not just capacity loss.

Two modeling phases:
- **Oracle**: controlled batteries only, uses EIS (`Re`, `Rct`) + `r_proxy`
- **Practical**: controlled + randomized merged, uses `r_proxy` only (field-deployable)

See `@notebooks/02_dataset_prep_summary.md` for feature list and leakage guards.

## Code Style

- Keep notebooks clean: one cell per logical step, markdown headers between sections
- Prefer `pathlib.Path` over `os.path`
- Use `random_state=42` for reproducibility in all sklearn and train/test splits
