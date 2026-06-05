# MLflow Experiment Tracking Guide

This project uses **MLflow** for experiment tracking. Everything is stored locally — no cloud account needed, no internet required.

---

## What Gets Tracked Automatically

Every time you run `run_all.py`, MLflow logs:

| Category | What's Logged |
|----------|---------------|
| **Experiment** | Name, start time, Python version, CLI flags (`--nested`, `--use-text`, `--tune`) |
| **Config** | All values from `src/config.py` (epochs, batch size, learning rate, dropout, etc.) |
| **Model Params** | Default hyperparameters (e.g. `C=1.0` for LogReg, `n_estimators=100` for XGBoost) |
| **Tuned Params** | Best params found by Optuna (sklearn) or grid search (DL) |
| **Per-Fold Metrics** | `accuracy`, `f1`, `precision`, `recall`, `roc_auc`, `pr_auc` for each outer fold |
| **Training Curves** | `train_loss` and `val_loss` per epoch (DL models only) |
| **Artifacts** | JSON result files, loss curves, model checkpoints (`.pt` files) |
| **Aggregated Results** | Mean ± std per model-phase, best model per phase |

---

## Quick Start

### 1. Run experiments (same as before)

```bash
# Single model with tuning
python run_all.py --models xgboost --phases 1 --nested --tune

# Full benchmark (11 models × 4 phases)
python run_all.py --models all --phases all --nested --use-text --tune
```

MLflow tracking is **automatic** — no extra flags needed.

### 2. View results in the MLflow UI

Open a **new terminal** and run:

```bash
# Using the virtual environment
.venv/Scripts/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open your browser at: **http://localhost:5000**

Or use the Python helper:

```bash
.venv/Scripts/python -c "from src.mlflow_tracker import tracker; tracker.launch_ui()"
```

### 3. What you'll see in the UI

- **Experiments** → Click "SAE_Prediction"
- **Runs table** → Each row is a model-phase combination (e.g. "XGBoost_phase1")
- **Expand a run** → See all parameters, metrics, and artifacts
- **Fold runs** → Nested under each parent run (click the ▶ arrow)
- **Compare runs** → Select multiple runs, click "Compare"
- **Charts** → ROC-AUC across folds, training curves, etc.

---

## Resume After a Crash

If your computer stops mid-run, **don't panic**. MLflow tracks what's already done.

### Check what's completed

```python
from src.mlflow_tracker import tracker

# See which folds are done for a model-phase
completed = tracker.get_completed_folds("XGBoost", "1")
print(f"Completed folds: {completed}")  # e.g. {0, 1, 2}
```

### Re-run (already-completed folds are skipped)

```bash
# Just re-run the same command — completed model-phases are auto-skipped
python run_all.py --models xgboost --phases 1 --nested --tune
```

You'll see:
```
  [SKIP] XGBoost | phase 1 already completed (5 folds)
```

### Run only missing models

```bash
# If some models failed, run just those
python run_all.py --models mlp cnn --phases 2 3 --nested --tune
```

---

## Project Structure

```
Project/
├── mlflow.db              ← SQLite database (all runs, params, metrics)
├── mlartifacts/           ← Artifacts (JSON files, checkpoints, plots)
├── mlruns/                ← Legacy MLflow directory (if using file store)
├── results/               ← JSON result files (still saved as before)
├── models/checkpoints/    ← PyTorch .pt files (still saved as before)
└── src/
    └── mlflow_tracker.py  ← Centralized tracking module
```

> **Note:** `mlflow.db` and `mlartifacts/` are in `.gitignore` by default. Don't commit them.

---

## Disable Tracking (if needed)

If you want to run without MLflow (e.g. for quick tests):

```python
from src.mlflow_tracker import tracker
tracker.disable()
```

Or set the environment variable:

```bash
set MLFLOW_DISABLE=true
python run_all.py --models xgboost --phases 1
```

---

## Query Results Programmatically

```python
import mlflow

# List all experiments
for exp in mlflow.search_experiments():
    print(f"{exp.experiment_id}: {exp.name}")

# Get all runs for an experiment
runs = mlflow.search_runs(experiment_ids=["1"])
print(runs[["tags.model_name", "tags.phase", "metrics.roc_auc"]])

# Find best run by ROC-AUC
best = runs.loc[runs["metrics.roc_auc"].idxmax()]
print(f"Best: {best['tags.model_name']} | ROC-AUC: {best['metrics.roc_auc']:.4f}")

# Get params for a specific run
run_id = best["run_id"]
run = mlflow.get_run(run_id)
print(run.data.params)
```

---

## Tips for Long Benchmarks

### Run in batches

Instead of running all 11 models × 4 phases at once (8-12 hours), split into chunks:

```bash
# Day 1: Tree-based models
python run_all.py --models xgboost lightgbm random_forest --phases all --nested --tune

# Day 2: Linear models
python run_all.py --models logistic_regression svm knn --phases all --nested --tune

# Day 3: Deep learning
python run_all.py --models mlp cnn rnn transformer ft_transformer --phases all --nested --tune
```

Each batch appends to the same experiment. The UI shows everything together.

### Keep the DB safe

The `mlflow.db` file is your experiment history. **Back it up** before major changes:

```bash
cp mlflow.db mlflow.db.backup.$(date +%Y%m%d)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `mlflow.db` gets too big | It's SQLite — compact with `VACUUM;` or delete old experiments |
| Can't start UI | Make sure port 5000 is free: `lsof -i :5000` (Linux/Mac) or `netstat -ano \| findstr 5000` (Windows) |
| UI shows empty experiment | The DB may have been recreated. Check `mlflow.db` exists and has data |
| Want to reset everything | Delete `mlflow.db` and `mlartifacts/` — tracking starts fresh |

---

## Architecture

```
run_all.py
    └── tracker.start_experiment("SAE_Prediction")
        └── For each model-phase:
            └── tracker.start_run("XGBoost", phase="1")
                ├── Log model defaults (params)
                ├── Log config (params)
                └── nested_cv_single_model()
                    ├── tune_model() → logs tuning results
                    ├── For each outer fold:
                    │   └── tracker.start_fold_run(...)
                    │       ├── Log fold params (threshold, input_dim)
                    │       ├── Log fold metrics (roc_auc, f1, ...)
                    │       └── Log artifact (JSON result file)
                    └── aggregate_and_print() → logs aggregated results
```

All tracking goes through `src/mlflow_tracker.py` — a single, clean API.
