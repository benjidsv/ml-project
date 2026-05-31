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
1. Baseline: linear regression, then random forest / gradient boosting (scikit-learn)
2. Advanced: PyTorch LSTM or Transformer for time-series prediction

Always report metrics (MAE, RMSE, R²) for every model and compare them in a table.

## Stack

- Data: `pandas`, `numpy`, `scipy`
- Baselines: `scikit-learn`
- Deep learning: `PyTorch`
- Notebooks: Jupyter (exploration) → `.py` scripts (final reproducible pipeline)

## Project Structure

```
data/          raw and processed datasets (gitignore large files)
notebooks/     exploratory and analysis notebooks
src/           reusable Python modules (preprocessing, features, models)
models/        saved model checkpoints
results/       metrics, plots, evaluation outputs
```

## Selected Datasets & Renewable Context

We will use two complementary datasets from the **NASA Prognostics Data Repository** to model battery degradation. This setup transitions from idealized laboratory environments to realistic, erratic renewable energy profiles.

Download: 
- controlled: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- randomized: 

The structure of both is detailed in `./dataset-structure.md`.

## Feature Mapping to the Problem Statement

To directly address the project goal of predicting battery **lifespan** and **efficiency** in a green energy context, we map the raw telemetry as follows:

* **Remaining Useful Life**
* **Operational Efficiency:** Evaluated by tracking the increase in internal resistance and tracking energy retention loss across sequential cycles (Voltage/Current integration over time).

## Code Style

- Keep notebooks clean: one cell per logical step, markdown headers between sections
- Prefer `pathlib.Path` over `os.path`
- Use `random_state=42` for reproducibility in all sklearn and train/test splits
