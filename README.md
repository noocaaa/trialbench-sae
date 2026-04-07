# TrialBench — Serious Adverse Event Prediction

> AI methods comparison for serious adverse event prediction in clinical trials using the TrialBench benchmark — AI4Science course project.

## Task

Predict whether a **serious adverse event** occurred (binary: 0/1) in a clinical trial, given multi-modal trial features. Dataset: **31,306** clinical trial records from [TrialBench](https://huyjj.github.io/Trialbench/).

**Reference:** Chen et al., *Scientific Data* 12:1564 (2025) — [https://doi.org/10.1038/s41597-025-05680-8](https://doi.org/10.1038/s41597-025-05680-8)

---
## Models compared

### Traditional ML
| Model | Script |
|---|---|
| Logistic Regression | `models/logistic_regression.py` |
| Random Forest | `models/random_forest.py` |
| XGBoost | `models/xgboost.py` |

### Deep Learning
| Model | Script |
|---|---|
| MLP | `models/mlp.py` |
| CNN | `models/cnn.py` |
| RNN / LSTM | `models/rnn.py` |
| GNN / Transformer | `models/gnn.py` |
---
## Evaluation metrics
All models are evaluated on the test set using:
- F1-Score
- PR-AUC
- ROC-AUC
- Precision
- Recall
- Accuracy
---
## Project structure

```
trialbench-sae-prediction/
├── data/                  # Downloaded datasets (git-ignored)
├── models/                # One file per model
├── notebooks/             # EDA and experiment notebooks
├── results/               # Saved metrics and outputs
├── evaluate.py            # Unified evaluation script
├── requirements.txt       # Python dependencies
└── README.md
```

---
## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/trialbench-sae-prediction.git
cd trialbench-sae-prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
```python
import trialbench

save_path = 'data/'
trialbench.function.download_all_data(save_path)
```

Or load directly per task:

```python
train_df, valid_df, test_df, num_classes, tabular_input_dim = \
    trialbench.function.load_data('adverse_event', 'II', data_format='df')
```

---
## Authors

- Noelia Carrasco — https://github.com/noocaaa/
- Fahad Alsofyani — https://github.com/falsofyan

AI4Science course — Dr. Tianfan Fu, Nanjing University