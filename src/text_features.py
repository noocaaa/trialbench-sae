"""
src/text_features.py — Text feature extraction for SAE prediction.

Extracts TF-IDF features from clinical trial text columns and concatenates
with tabular features. Designed to integrate seamlessly with existing pipeline.

Usage:
    from src.text_features import extract_text_features, TEXT_COLUMNS_TO_USE
    X_train_text, X_test_text = extract_text_features(
        X_train_raw, X_test_raw, phase="1", max_features=100, verbose=True
    )
    # Then concatenate: X_train_combined = np.hstack([X_train_tabular, X_train_text])
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Text columns we want to extract features from ─────────────────
# Selected based on: low null rate, high uniqueness, clinical relevance
TEXT_COLUMNS_TO_USE = [
    "brief_title",                          # Short trial title — very informative
    "brief_summary/textblock",              # Full trial summary — most informative
    "condition",                            # Disease/condition being studied
    "eligibility/criteria/textblock",       # Inclusion/exclusion criteria
]


# ── TF-IDF parameters ─────────────────────────────────────────────
# Conservative defaults to avoid overfitting on small datasets
TFIDF_MAX_FEATURES = 100      # Features per text column
TFIDF_MIN_DF = 2              # Ignore terms that appear in < 2 documents
TFIDF_MAX_DF = 0.95           # Ignore terms that appear in > 95% of documents
TFIDF_NGRAM_RANGE = (1, 2)    # Unigrams + bigrams


def _clean_text(text):
    """
    Clean raw text for TF-IDF.
    
    Handles:
    - NaN/None -> empty string
    - List-like strings (e.g., "['A', 'B']") -> join with spaces
    - Extra whitespace normalization
    """
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # Handle list-like strings: "['Neoplasms', 'Cancer']" -> "Neoplasms Cancer"
    if text.startswith("[") and text.endswith("]"):
        try:
            import ast
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                text = " ".join(str(item) for item in parsed if item)
        except (ValueError, SyntaxError):
            pass  # Keep as-is if parsing fails
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    return text


def _build_text_column(df, text_cols):
    """
    Combine multiple text columns into a single document per row.
    
    Each column is prefixed with its name so TF-IDF can distinguish
    the same word in different contexts (e.g., "title: cancer" vs "condition: cancer").
    """
    documents = []
    for idx in range(len(df)):
        parts = []
        for col in text_cols:
            if col in df.columns:
                raw = df.iloc[idx][col]
                cleaned = _clean_text(raw)
                if cleaned:
                    # Prefix with column name for context
                    prefix = col.replace("/", "_").replace("-", "_")
                    parts.append(f"{prefix} {cleaned}")
        documents.append(" ".join(parts) if parts else "no_text")
    return documents


def extract_text_features(
    X_train_raw,
    X_test_raw,
    phase="?",
    text_cols=None,
    max_features=None,
    min_df=None,
    max_df=None,
    ngram_range=None,
    verbose=False,
):
    """
    Extract TF-IDF features from text columns and return as numpy arrays.

    Parameters
    ----------
    X_train_raw : pd.DataFrame — raw training features (BEFORE _preprocess drops text)
    X_test_raw  : pd.DataFrame — raw test features (BEFORE _preprocess drops text)
    phase       : str — phase label for verbose output
    text_cols   : list — text columns to use (default: TEXT_COLUMNS_TO_USE)
    max_features: int — max TF-IDF features (default: TFIDF_MAX_FEATURES)
    min_df      : int — min document frequency (default: TFIDF_MIN_DF)
    max_df      : float — max document frequency ratio (default: TFIDF_MAX_DF)
    ngram_range : tuple — (min_n, max_n) for n-grams (default: (1, 2))
    verbose     : bool — print summary

    Returns
    -------
    X_train_text, X_test_text : np.ndarray — TF-IDF feature matrices
                                shape: (n_samples, max_features)
    """
    text_cols = text_cols or TEXT_COLUMNS_TO_USE
    max_features = max_features if max_features is not None else TFIDF_MAX_FEATURES
    min_df = min_df if min_df is not None else TFIDF_MIN_DF
    max_df = max_df if max_df is not None else TFIDF_MAX_DF
    ngram_range = ngram_range if ngram_range is not None else TFIDF_NGRAM_RANGE

    # Check which columns actually exist
    available_cols = [c for c in text_cols if c in X_train_raw.columns]
    if not available_cols:
        if verbose:
            print(f"  Phase {phase} — No text columns found, returning empty features")
        return (
            np.zeros((len(X_train_raw), 0), dtype=np.float32),
            np.zeros((len(X_test_raw), 0), dtype=np.float32),
        )

    # Build combined documents
    train_docs = _build_text_column(X_train_raw, available_cols)
    test_docs = _build_text_column(X_test_raw, available_cols)

    # Fit TF-IDF on training data only (no leakage)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
    )

    X_train_text = vectorizer.fit_transform(train_docs)
    X_test_text = vectorizer.transform(test_docs)

    # Convert to dense numpy array (tabular models expect dense input)
    X_train_text = X_train_text.toarray().astype(np.float32)
    X_test_text = X_test_text.toarray().astype(np.float32)

    if verbose:
        print(f"\n  Phase {phase} — Text feature extraction:")
        print(f"    Text columns used : {available_cols}")
        print(f"    Train documents   : {len(train_docs)}")
        print(f"    Vocabulary size   : {len(vectorizer.vocabulary_)}")
        print(f"    TF-IDF features   : {X_train_text.shape[1]}")
        # Show top terms by mean TF-IDF score
        mean_scores = X_train_text.mean(axis=0)
        top_indices = np.argsort(mean_scores)[-10:][::-1]
        feature_names = vectorizer.get_feature_names_out()
        top_terms = [feature_names[i] for i in top_indices]
        print(f"    Top terms         : {', '.join(top_terms)}")

    return X_train_text, X_test_text


