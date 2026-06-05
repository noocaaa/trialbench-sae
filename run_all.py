"""
Run all models for SAE Prediction.

Supports both simple train/test split evaluation and nested cross-validation.

Usage:
    # Run all models with simple train/test split (default)
    python run_all.py

    # Run all models with nested cross-validation
    python run_all.py --nested

    # Run specific models
    python run_all.py --models mlp cnn random_forest xgboost

    # Run specific phases
    python run_all.py --models mlp --phases 1 2

    # Clear old results and rerun
    python run_all.py --clear
"""
import argparse
import importlib
import traceback

from src.data_loader import set_seed
from src.utils import clear_results, print_results_table, get_best_model

# Nested CV imports
from src.nested_cv import nested_cv_single_model
from src.aggregate_results import aggregate_and_print
from src import config

# Model imports for factory functions (nested CV)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from models.mlp import MLP
from models.cnn import CNN
from models.rnn import RNN
from models.transformer import Transformer


PHASES = ["1", "2", "3", "4"]

# Model registry for simple train/test split: name -> (module_path, function_name)
SIMPLE_MODELS = {
    # Deep Learning models (PyTorch)
    "mlp":                 ("models.mlp",                 "run"),
    "cnn":                 ("models.cnn",                 "run"),
    "rnn":                 ("models.rnn",                 "run"),
    "transformer":         ("models.transformer",         "run"),

    # Classical ML models (sklearn)
    "logistic_regression": ("models.logistic_regression", "run"),
    "random_forest":       ("models.random_forest",       "run"),
    "svm":                 ("models.svm",                 "run"),
    "knn":                 ("models.knn",                 "run"),
    "xgboost":             ("models.xgboost_model",       "run"),
}

# Model registry for nested CV: name -> (is_tree, is_dl, display_name)
NESTED_MODELS = {
    "logistic_regression": (False, False, "LogisticRegression"),
    "random_forest":       (True,  False, "RandomForest"),
    "xgboost":             (True,  False, "XGBoost"),
    "svm":                 (False, False, "SVM"),
    "knn":                 (False, False, "KNN"),
    "mlp":                 (False, True,  "MLP"),
    "cnn":                 (False, True,  "CNN"),
    "rnn":                 (False, True,  "RNN"),
    "transformer":         (False, True,  "Transformer"),
}


def _make_factory(name):
    """Create a factory function for nested CV."""
    if name == "logistic_regression":
        return lambda **kwargs: LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=42
        )
    elif name == "random_forest":
        return lambda **kwargs: RandomForestClassifier(
            n_estimators=100, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, random_state=42, n_jobs=-1,
        )
    elif name == "xgboost":
        return lambda **kwargs: XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    elif name == "svm":
        return lambda **kwargs: CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=10000, random_state=42),
            method="sigmoid", cv=2,
        )
    elif name == "knn":
        return lambda **kwargs: KNeighborsClassifier(
            n_neighbors=5, weights="distance", n_jobs=-1
        )
    elif name == "mlp":
        return lambda **kwargs: MLP(kwargs["input_dim"])
    elif name == "cnn":
        return lambda **kwargs: CNN(kwargs["input_dim"])
    elif name == "rnn":
        return lambda **kwargs: RNN(kwargs["input_dim"])
    elif name == "transformer":
        return lambda **kwargs: Transformer(kwargs["input_dim"])
    else:
        raise ValueError(f"Unknown model: {name}")


def run_simple_model(name, phase):
    """Run a single model using simple train/test split."""
    module_path, fn_name = SIMPLE_MODELS[name]
    try:
        module = importlib.import_module(module_path)
        getattr(module, fn_name)(phase)
    except Exception as e:
        print(f"  [ERROR] {name} | phase {phase}: {e}")
        traceback.print_exc()


def run_nested_model(name, phase):
    """Run a single model using nested cross-validation."""
    is_tree, is_dl, display_name = NESTED_MODELS[name]
    factory_fn = _make_factory(name)
    try:
        nested_cv_single_model(
            model_fn=factory_fn,
            phase=phase,
            model_name=display_name,
            is_tree=is_tree,
            is_dl=is_dl,
            outer_folds=config.OUTER_FOLDS,
            inner_folds=config.INNER_FOLDS,
            verbose=True,
        )
    except Exception as e:
        print(f"  [ERROR] {name} | phase {phase}: {e}")
        traceback.print_exc()


def main(models, phases, clear, nested):
    """Run all requested models on all requested phases."""
    set_seed()

    if clear:
        clear_results()

    for phase in phases:
        print(f"\n{'='*60}\n  PHASE {phase}\n{'='*60}")
        for name in models:
            if nested:
                print(f"\n>> Running {name} (nested CV)...")
                run_nested_model(name, phase)
            else:
                print(f"\n>> Running {name}...")
                run_simple_model(name, phase)

    print(f"\n{'='*60}\n  FINAL RESULTS\n{'='*60}")
    if nested:
        aggregate_and_print()
    else:
        print_results_table()
        get_best_model(metric="f1")
        get_best_model(metric="roc_auc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all models for SAE prediction."
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=list(SIMPLE_MODELS.keys()),
        default=list(SIMPLE_MODELS.keys()),
        help="Models to run (default: all 9 models)"
    )
    parser.add_argument(
        "--phases", nargs="+",
        choices=PHASES,
        default=PHASES,
        help="Trial phases to evaluate (default: all 4 phases)"
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Clear previous results before running"
    )
    parser.add_argument(
        "--nested", action="store_true",
        help="Use nested cross-validation (5 outer x 3 inner folds) instead of simple train/test split"
    )
    args = parser.parse_args()

    main(args.models, args.phases, args.clear, args.nested)
