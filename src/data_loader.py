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


set_seed()


def load_phase(phase, data_dir=None, verbose=False):
    """
    Load train/test splits for a given phase.

    Parameters
    ----------
    phase   : str  — "1", "2", "3", or "4"
    data_dir: str  — path to dataset root. If None uses data/ in project root.
    verbose : bool — if True prints feature selection summary

    Returns
    -------
    X_train, X_test : np.ndarray — preprocessed feature matrices
    y_train, y_test : np.ndarray — binary labels (0/1)
    pos_weight      : float      — neg/pos ratio for BCEWithLogitsLoss
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA

    base = os.path.join(data_dir, f"Phase{phase}")

    X_train = pd.read_csv(os.path.join(base, "train_x.csv"))
    y_train = pd.read_csv(os.path.join(base, "train_y.csv"))
    X_test  = pd.read_csv(os.path.join(base, "test_x.csv"))
    y_test  = pd.read_csv(os.path.join(base, "test_y.csv"))

    y_train = y_train["Y/N"].values.astype(int)
    y_test  = y_test["Y/N"].values.astype(int)

    X_train, X_test = _preprocess(X_train, X_test, phase=phase, verbose=verbose)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    pos_weight = float(neg / pos)

    return X_train, X_test, y_train, y_test, pos_weight


def _preprocess(X_train, X_test, phase="?", verbose=False):
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

    # ── Step 4: encode categorical columns → numeric ───────────────
    for col in X_train.select_dtypes(include="object").columns:
        X_train[col] = X_train[col].astype("category").cat.codes
        X_test[col]  = X_test[col].astype("category").cat.codes

    # ── Step 5: impute remaining missing values with median ────────
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    # ── Step 6: standardize ────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

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