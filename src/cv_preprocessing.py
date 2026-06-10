"""
src/cv_preprocessing.py — Leakage-free preprocessing for cross-validation.

All statistics (null rates, imputer, scaler, encoders, TF-IDF) are fit on the
training fold only and applied to both train and test. This prevents data
leakage from the test fold into the training pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from src.data_loader import TEXT_COLS, ZERO_VARIANCE_COLS, NULL_THRESHOLD


def preprocess_cv(X_train, X_test, phase="?", for_tree=False, verbose=False, use_text=False,
                    precomputed_train_docs=None, precomputed_test_docs=None):
    """
    Preprocess train/test splits for cross-validation with NO data leakage.

    All statistics are computed on X_train only and applied to both sets.

    Parameters
    ----------
    X_train  : pd.DataFrame — training features for this fold
    X_test   : pd.DataFrame — test features for this fold
    phase    : str  — phase label for verbose output
    for_tree : bool — if True, use OrdinalEncoder + no standardization
    verbose  : bool — if True, print feature selection summary
    use_text : bool — if True, append TF-IDF features from text columns
    precomputed_train_docs : list — pre-built document strings for train (optional, for caching)
    precomputed_test_docs  : list — pre-built document strings for test (optional, for caching)

    Returns
    -------
    X_train_np, X_test_np : np.ndarray — preprocessed feature matrices
    """
    # Work on copies to avoid modifying caller's data
    X_tr = X_train.copy()
    X_te = X_test.copy()
    n_original = len(X_tr.columns)

    # ── Extract text features BEFORE dropping text columns ──────────
    X_train_text = X_test_text = None
    if use_text:
        if precomputed_train_docs is not None and precomputed_test_docs is not None:
            # Use pre-built documents (cached from full dataset)
            from src.text_features import _make_vectorizer
            
            vectorizer = _make_vectorizer()
            X_train_text = vectorizer.fit_transform(precomputed_train_docs).toarray().astype(np.float32)
            X_test_text = vectorizer.transform(precomputed_test_docs).toarray().astype(np.float32)
        else:
            from src.text_features import extract_text_features
            X_train_text, X_test_text = extract_text_features(
                X_tr, X_te, phase=phase, verbose=verbose
            )

    # ── Step 1: drop ID column and text columns ───────────────────
    drop_cols = ["Unnamed: 0"] + [c for c in TEXT_COLS if c in X_tr.columns]
    X_tr = X_tr.drop(columns=drop_cols, errors="ignore")
    X_te = X_te.drop(columns=drop_cols, errors="ignore")
    n_after_text = len(X_tr.columns)

    # ── Step 2: drop zero-variance columns (same value always) ────
    zero_var = [c for c in ZERO_VARIANCE_COLS if c in X_tr.columns]
    X_tr = X_tr.drop(columns=zero_var, errors="ignore")
    X_te = X_te.drop(columns=zero_var, errors="ignore")
    n_after_zero = len(X_tr.columns)

    # ── Step 3: drop columns with >80% missing values ─────────────
    # Computed on train set only to avoid data leakage
    null_rates = X_tr.isnull().mean()
    high_null = null_rates[null_rates > NULL_THRESHOLD].index.tolist()
    X_tr = X_tr.drop(columns=high_null, errors="ignore")
    X_te = X_te.drop(columns=high_null, errors="ignore")
    n_after_null = len(X_tr.columns)

    # ── Step 4: encode categorical columns ─────────────────────────
    cat_cols = X_tr.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        if for_tree:
            # Tree models: ordinal encoding (no dummies, no scaling later)
            for col in cat_cols:
                fill_val = (
                    X_tr[col].mode().iloc[0]
                    if not X_tr[col].mode().empty
                    else "missing"
                )
                X_tr[col] = X_tr[col].fillna(fill_val)
                X_te[col] = X_te[col].fillna(fill_val)
            oe = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            X_tr[cat_cols] = oe.fit_transform(X_tr[cat_cols])
            X_te[cat_cols] = oe.transform(X_te[cat_cols])
        else:
            # Other models: one-hot encoding
            X_tr = pd.get_dummies(X_tr, columns=cat_cols, dummy_na=False)
            X_te = pd.get_dummies(X_te, columns=cat_cols, dummy_na=False)
            # Align test columns to train (handle unseen categories)
            X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

    # ── Step 5: impute remaining missing values with median ────────
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_tr)
    X_te = imputer.transform(X_te)

    # ── Step 6: standardize (skip for tree-based models) ───────────
    if not for_tree:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    # ── Step 7: concatenate text features if requested ─────────────
    if use_text and X_train_text is not None and X_train_text.shape[1] > 0:
        X_tr = np.hstack([X_tr, X_train_text])
        X_te = np.hstack([X_te, X_test_text])

    # ── Summary ───────────────────────────────────────────────────
    if verbose:
        print(f"\n  Phase {phase} - CV feature selection summary:")
        print(
            f"    Original columns     : {n_original}"
        )
        print(
            f"    After dropping text  : {n_after_text}  "
            f"(-{n_original - n_after_text} text cols)"
        )
        print(
            f"    After zero-variance  : {n_after_zero}  "
            f"(-{n_after_text - n_after_zero} constant cols)"
        )
        print(
            f"    After high nulls     : {n_after_null}  "
            f"(-{n_after_zero - n_after_null} cols >{NULL_THRESHOLD*100:.0f}% null)"
        )
        final_count = X_tr.shape[1]
        if use_text and X_train_text is not None:
            tab_count = final_count - X_train_text.shape[1]
            print(f"    Tabular features     : {tab_count}")
            print(f"    Text features        : {X_train_text.shape[1]}")
            print(f"    Final feature count  : {final_count}")
        else:
            print(f"    Final feature count  : {final_count}")
        if high_null:
            print(f"    Dropped (high null)  : {high_null}")

    return X_tr.astype(np.float32), X_te.astype(np.float32)
