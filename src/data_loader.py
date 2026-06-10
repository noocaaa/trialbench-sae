import os
import random
import numpy as np
import pandas as pd
import torch
# ── Text columns — dropped because models can't process raw text ──
TEXT_COLS = [
    "brief_summary/textblock",
    "brief_title",
    "condition",
    "condition_browse/mesh_term",
    "eligibility/criteria/textblock",
    "intervention/description",
    "intervention/intervention_name",
    "intervention_browse/mesh_term",
    "keyword",
    "location/facility/address/city",
    "responsible_party/responsible_party_type",
    "smiless",
    "icdcode",
    "study_design_info/intervention_model_description",
    "study_design_info/masking_description",
    "patient_data/sharing_ipd",
]

# ── Columns with zero variance — same value for every row ─────────
ZERO_VARIANCE_COLS = [
    "phase",        # always same within a phase dataset
    "study_type",   # always "Interventional"
]

# ── High null threshold — drop columns above this % of nulls ──────
NULL_THRESHOLD = 0.80  # drop columns with >80% missing values

SEED = 42

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATA = os.path.join(_PROJECT_ROOT, "data", "serious-adverse-event-forecasting")

# ── CSV Cache ─────────────────────────────────────────────────────
# Cache DataFrames in memory to avoid re-reading CSVs multiple times
# (especially important for nested CV with 5+ folds per phase)
_CSV_CACHE = {}


def _read_csv_cached(path):
    """Read CSV with in-memory caching."""
    if path not in _CSV_CACHE:
        _CSV_CACHE[path] = pd.read_csv(path)
    return _CSV_CACHE[path]


def clear_csv_cache():
    """Clear the CSV cache (useful before re-running experiments)."""
    global _CSV_CACHE
    _CSV_CACHE = {}


def load_phase(phase, data_dir=None, verbose=False, for_tree=False, use_text=False, text_max_features=100):
    """
    Load train/test splits for a given phase.

    Parameters
    ----------
    phase    : str  — "1", "2", "3", or "4"
    data_dir : str  — path to dataset root. If None uses data/ in project root.
    verbose  : bool — if True prints feature selection summary
    for_tree : bool — if True, use OrdinalEncoder + no scaling for tree-based models
    use_text : bool — if True, append TF-IDF features from text columns
    text_max_features : int — max TF-IDF features per text column (default: 100)

    Returns
    -------
    X_train, X_test : np.ndarray — preprocessed feature matrices
    y_train, y_test : np.ndarray — binary labels (0/1)
    pos_weight      : float      — neg/pos ratio for BCEWithLogitsLoss
    """
    if phase not in ("1", "2", "3", "4"):
        raise ValueError(f"phase must be one of '1', '2', '3', '4', got {phase!r}")

    if data_dir is None:
        data_dir = _DEFAULT_DATA

    base = os.path.join(data_dir, f"Phase{phase}")

    if not os.path.isdir(base):
        raise FileNotFoundError(f"Dataset directory not found: {base}")

    train_x_path = os.path.join(base, "train_x.csv")
    train_y_path = os.path.join(base, "train_y.csv")
    test_x_path  = os.path.join(base, "test_x.csv")
    test_y_path  = os.path.join(base, "test_y.csv")

    for p in (train_x_path, train_y_path, test_x_path, test_y_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    X_train_raw = _read_csv_cached(train_x_path)
    y_train = _read_csv_cached(train_y_path)
    X_test_raw  = _read_csv_cached(test_x_path)
    y_test  = _read_csv_cached(test_y_path)

    y_train = y_train["Y/N"].values.astype(int)
    y_test  = y_test["Y/N"].values.astype(int)

    if y_train.ndim != 1 or y_test.ndim != 1:
        raise ValueError("y_train and y_test must be 1-dimensional")

    if len(set(np.unique(y_train)) - {0, 1}) > 0:
        raise ValueError("y_train contains non-binary values (must be 0 or 1)")

    # ── Preprocess using unified pipeline (same as nested CV) ──
    from src.cv_preprocessing import preprocess_cv
    X_train, X_test = preprocess_cv(
        X_train_raw, X_test_raw, phase=phase, for_tree=for_tree, verbose=verbose, use_text=use_text
    )

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"X_train/y_train shape mismatch: {X_train.shape[0]} vs {y_train.shape[0]}")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"X_test/y_test shape mismatch: {X_test.shape[0]} vs {y_test.shape[0]}")

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    pos_weight = float(neg / pos) if pos > 0 else 1.0

    return X_train, X_test, y_train, y_test, pos_weight


