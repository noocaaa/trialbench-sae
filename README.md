# TrialBench — Serious Adverse Event Prediction

> Comparison of 9 AI methods for predicting serious adverse events in clinical trials.

**Reference:** Chen et al., *Scientific Data* 12:1564 (2025) — [https://doi.org/10.1038/s41597-025-05680-8](https://doi.org/10.1038/s41597-025-05680-8)

---
## Task

**Binary classification:** predict whether a serious adverse event (SAE) occurred in a clinical trial given tabular trial features (intervention type, masking, enrollment, eligibility, etc.).

- **Dataset:** 31,306 clinical trial records from [TrialBench](https://huyjj.github.io/Trialbench/) — 17,916 available across 4 phases
- **Splits:** 4 phases (Phase 1–4), each with `train_x/y.csv` and `test_x/y.csv`
- **Label:** `Y/N` column in `*_y.csv` (1 = SAE occurred, 0 = did not)
- **Features used:** ~37 tabular features after preprocessing (text and high-null columns excluded — see `src/data_loader.py`)

---

## Project structure

```
trialbench-sae-prediction/
│
├── data/                              # Datasets — git-ignored
│   └── serious-adverse-event-forecasting/
│       ├── Phase1/
│       │   ├── train_x.csv
│       │   ├── train_y.csv
│       │   ├── test_x.csv
│       │   └── test_y.csv
│       └── Phase2–4/ ...
│
├── src/                               # Shared utilities — read before adding models
│   ├── config.py                      # ← All training hyperparameters (epochs, lr, etc.)
│   ├── train.py                       # ← Shared training loop (PyTorch models use this)
│   ├── data_loader.py                 # Data loading + preprocessing + global seed
│   ├── evaluate.py                    # Shared evaluation — MUST call this in every model
│   ├── plot_results.py                # Interactive results dashboard
│   └── utils.py                       # Helper functions for results
│
├── models/                            # One file per model — only architecture + run()
│   ├── mlp.py                         # Multi-Layer Perceptron
│   ├── cnn.py                         # 1D Convolutional Network
│   ├── rnn.py                         # LSTM Recurrent Network
│   ├── transformer.py                 # Transformer Encoder
│   ├── logistic_regression.py         # Logistic Regression
│   ├── random_forest.py               # Random Forest
│   ├── svm.py                         # Support Vector Machine
│   ├── knn.py                         # K-Nearest Neighbors
│   └── xgboost_model.py               # XGBoost Gradient Boosting
│
├── apps/
│   └── eda.py                         # Exploratory Data Analysis — 4-tab dashboard
│
├── results/                           # Auto-generated — one JSON per model × phase
│
├── run_all.py                         # Entry point — run any combination of models
├── sanity_check.py                    # Automated validation of all results
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/noocaaa/trialbench-sae-prediction.git
cd trialbench-sae-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Mac / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

```python
import trialbench
trialbench.function.download_all_data("data/")
```

---

## Usage

```bash
# Run deep learning models
python run_all.py --models mlp cnn rnn transformer

# Run classical ML models
python run_all.py --models logistic_regression random_forest svm knn

# Run specific phases only
python run_all.py --models mlp --phases 1 2

# Clear old results and rerun everything
python run_all.py --clear

# Results dashboard (interactive)
python src/plot_results.py        # → http://127.0.0.1:8050

# EDA dashboard
python apps/eda.py                # → http://127.0.0.1:8052

# Validate all results
python sanity_check.py
```

---

## Training configuration (`src/config.py`)

All deep learning models use the **same hyperparameters** for fair comparison:

| Parameter | Value | Why |
|---|---|---|
| `EPOCHS` | 100 | Max epochs; early stopping usually kicks in first |
| `LR` | 3e-4 | Lower than default — needed for Transformer stability |
| `BATCH_SIZE` | 64 | Standard, matches TrialBench paper |
| `OPTIMIZER` | AdamW | Better generalization than Adam via weight decay |
| `WEIGHT_DECAY` | 1e-4 | Regularization |
| `SCHEDULER` | CosineAnnealingLR | Smooth lr decay for better convergence |
| `GRAD_CLIP` | 1.0 | Prevents exploding gradients (critical for Transformer) |
| `VAL_SPLIT` | 0.125 | 12.5% of training data held out for validation |
| `PATIENCE` | 5 | Early stopping: stop if val loss does not improve for 5 epochs |
| `TUNE_THRESHOLD` | True | Find optimal decision threshold on validation set |

---

## `src/evaluate.py` — shared evaluation

Every model call this at the end of `run()`:

```python
from src.evaluate import evaluate

# sklearn
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# PyTorch
y_prob = torch.sigmoid(logits).detach().numpy()
y_pred = (y_prob >= 0.5).astype(int)

evaluate(y_test, y_pred, y_prob, model_name="Model", phase=phase)
```

Saves to `results/YourModel_phase.json`:
```json
{
  "model": "Model", "phase": "1",
  "accuracy": 0.78, "f1": 0.75, "precision": 0.80,
  "recall": 0.71, "roc_auc": 0.85, "pr_auc": 0.82,
  "y_pred": [...],
  "y_test": [...]
}
```

---

## Results

Results are saved automatically to `results/` after each run.

> Run `python src/plot_results.py` for the interactive dashboard.
> 
> Run `python sanity_check.py` for the full validation report (dataset stats, metric consistency, dummy baseline comparison, confusion matrices, and loss curve analysis).


**Class balance per phase:**

| Phase | Trials | SAE rate | Balance | pos_weight |
|---|---|---|---|---|
| 1 | 2,014 | 43.4% | Balanced | 1.23 |
| 2 | 8,116 | 74.6% | Imbalanced | 0.39 |
| 3 | 4,840 | 84.7% | Very imbalanced | 0.18 |
| 4 | 2,946 | 38.5% | Balanced | 1.67 |

> **⚠️ Interpreting metrics:** Phases 2 & 3 are heavily imbalanced. A dummy classifier that always predicts "SAE" achieves F1 ≈ 0.86–0.92 in these phases. **Use ROC-AUC as the primary metric** — it is invariant to class imbalance and a dummy always scores 0.5. Run `python sanity_check.py` to see dummy baselines and verify your models are genuinely learning.

---

## Reproducibility

- `seed=42` set globally in `src/data_loader.py` on import
- Covers Python `random`, NumPy, and PyTorch
- No extra setup needed in model files

---

## Authors

- **Noelia Carrasco** — Deep Learning models (MLP, CNN, RNN, Transformer) + EDA + data pipeline + dashboards
- **Fahad Alsofyani** — Classical ML models (LR, RF, SVM, KNN) + Report

AI4Science course — Dr. Tianfan Fu, Nanjing University