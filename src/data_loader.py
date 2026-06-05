import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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


def set_seed(seed=SEED):
    """Set all random seeds globally for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_phase(phase, data_dir=None, verbose=False, for_tree=False):
    """
    Load train/test splits for a given phase.

    Parameters
    ----------
    phase    : str  — "1", "2", "3", or "4"
    data_dir : str  — path to dataset root. If None uses data/ in project root.
    verbose  : bool — if True prints feature selection summary
    for_tree : bool — if True, use OrdinalEncoder + no scaling for tree-based models

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

    X_train = pd.read_csv(train_x_path)
    y_train = pd.read_csv(train_y_path)
    X_test  = pd.read_csv(test_x_path)
    y_test  = pd.read_csv(test_y_path)

    y_train = y_train["Y/N"].values.astype(int)
    y_test  = y_test["Y/N"].values.astype(int)

    if y_train.ndim != 1 or y_test.ndim != 1:
        raise ValueError("y_train and y_test must be 1-dimensional")

    if len(set(np.unique(y_train)) - {0, 1}) > 0:
        raise ValueError("y_train contains non-binary values (must be 0 or 1)")

    X_train, X_test = _preprocess(
        X_train, X_test, phase=phase, verbose=verbose, for_tree=for_tree
    )

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"X_train/y_train shape mismatch: {X_train.shape[0]} vs {y_train.shape[0]}")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"X_test/y_test shape mismatch: {X_test.shape[0]} vs {y_test.shape[0]}")

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    pos_weight = float(neg / pos) if pos > 0 else 1.0

    return X_train, X_test, y_train, y_test, pos_weight


def _preprocess(X_train, X_test, phase="?", verbose=False, for_tree=False):
    n_original = len(X_train.columns)

    # ── Step 1: drop ID column and text columns ───────────────────
    drop_cols = ["Unnamed: 0"] + [c for c in TEXT_COLS if c in X_train.columns]
    X_train = X_train.drop(columns=drop_cols, errors="ignore")
    X_test  = X_test.drop(columns=drop_cols, errors="ignore")
    n_after_text = len(X_train.columns)

    # ── Step 2: drop zero-variance columns (same value always) ────
    zero_var = [c for c in ZERO_VARIANCE_COLS if c in X_train.columns]
    X_train = X_train.drop(columns=zero_var, errors="ignore")
    X_test  = X_test.drop(columns=zero_var, errors="ignore")
    n_after_zero = len(X_train.columns)

    # ── Step 3: drop columns with >80% missing values ─────────────
    # Computed on train set only to avoid data leakage
    null_rates  = X_train.isnull().mean()
    high_null   = null_rates[null_rates > NULL_THRESHOLD].index.tolist()
    X_train = X_train.drop(columns=high_null, errors="ignore")
    X_test  = X_test.drop(columns=high_null, errors="ignore")
    n_after_null = len(X_train.columns)

    # ── Step 4: encode categorical columns ─────────────────────────
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        if for_tree:
            # Tree models: ordinal encoding (no dummies, no scaling later)
            from sklearn.preprocessing import OrdinalEncoder
            for col in cat_cols:
                fill_val = X_train[col].mode().iloc[0] if not X_train[col].mode().empty else "missing"
                X_train[col] = X_train[col].fillna(fill_val)
                X_test[col] = X_test[col].fillna(fill_val)
            oe = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            X_train[cat_cols] = oe.fit_transform(X_train[cat_cols])
            X_test[cat_cols] = oe.transform(X_test[cat_cols])
        else:
            # Other models: one-hot encoding
            X_train = pd.get_dummies(X_train, columns=cat_cols, dummy_na=False)
            X_test = pd.get_dummies(X_test, columns=cat_cols, dummy_na=False)
            # Align test columns to train (handle unseen categories in either split)
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # ── Step 5: impute remaining missing values with median ────────
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    # ── Step 6: standardize (skip for tree-based models) ───────────
    if not for_tree:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # ── Summary ───────────────────────────────────────────────────
    if verbose:
        print(f"\n  Phase {phase} — Feature selection summary:")
        print(f"    Original columns     : {n_original}")
        print(f"    After dropping text  : {n_after_text}  (-{n_original - n_after_text} text cols)")
        print(f"    After zero-variance  : {n_after_zero}  (-{n_after_text - n_after_zero} constant cols)")
        print(f"    After high nulls     : {n_after_null}  (-{n_after_zero - n_after_null} cols >{NULL_THRESHOLD*100:.0f}% null)")
        print(f"    Final feature count  : {n_after_null}")
        if high_null:
            print(f"    Dropped (high null)  : {high_null}")

    return X_train.astype(np.float32), X_test.astype(np.float32)