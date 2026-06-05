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

    # Run with text features (TF-IDF from trial descriptions)
    python run_all.py --models mlp --use-text

    # Run with hyperparameter tuning (sklearn models only)
    python run_all.py --models xgboost random_forest --nested --tune

    # Run specific phases
    python run_all.py --models mlp --phases 1 2

    # Clear old results and rerun
    python run_all.py --clear
"""
import argparse
import importlib
import traceback
import warnings

# Silenciar warnings de sklearn sobre feature names (nested CV usa np.ndarray)
warnings.filterwarnings("ignore", category=UserWarning, message=".*feature names.*")

from src.data_loader import set_seed
from src.utils import clear_results, print_results_table, get_best_model

# Nested CV imports
from src.nested_cv import nested_cv_single_model
from src.aggregate_results import aggregate_and_print
from src import config
from src.mlflow_tracker import tracker

# Model imports for factory functions (nested CV)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from models.mlp import MLP
from models.cnn import CNN
from models.rnn import RNN
from models.transformer import Transformer
from models.ft_transformer import FTTransformer


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
    "lightgbm":            ("models.lightgbm_model",      "run"),
    "ft_transformer":      ("models.ft_transformer",      "run"),


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
    "lightgbm":            (True,  False, "LightGBM"),
    "ft_transformer":      (False, True,  "FT-Transformer"),
}


def _make_factory(name):
    """Create a factory function for nested CV."""
    if name == "logistic_regression":
        def _logreg_factory(**kwargs):
            C = kwargs.get("C", 1.0)
            return LogisticRegression(
                C=C, max_iter=1000, solver="lbfgs", random_state=42
            )
        return _logreg_factory
    elif name == "random_forest":
        def _rf_factory(**kwargs):
            n_estimators = kwargs.get("n_estimators", 100)
            max_depth = kwargs.get("max_depth", None)
            min_samples_split = kwargs.get("min_samples_split", 2)
            return RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=1, random_state=42, n_jobs=-1,
            )
        return _rf_factory
    elif name == "xgboost":
        def _xgb_factory(**kwargs):
            n_estimators = kwargs.get("n_estimators", 100)
            max_depth = kwargs.get("max_depth", 6)
            learning_rate = kwargs.get("learning_rate", 0.1)
            subsample = kwargs.get("subsample", 0.8)
            colsample_bytree = kwargs.get("colsample_bytree", 0.8)
            return XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, subsample=subsample,
                colsample_bytree=colsample_bytree, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            )
        return _xgb_factory
    elif name == "svm":
        def _svm_factory(**kwargs):
            C = kwargs.get("C", 1.0)
            return CalibratedClassifierCV(
                LinearSVC(C=C, max_iter=10000, random_state=42),
                method="sigmoid", cv=3,
            )
        return _svm_factory
    elif name == "knn":
        def _knn_factory(**kwargs):
            n_neighbors = kwargs.get("n_neighbors", 5)
            weights = kwargs.get("weights", "distance")
            return KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=weights, n_jobs=-1
            )
        return _knn_factory
    elif name == "mlp":
        return lambda **kwargs: MLP(kwargs["input_dim"])
    elif name == "cnn":
        return lambda **kwargs: CNN(kwargs["input_dim"])
    elif name == "rnn":
        return lambda **kwargs: RNN(kwargs["input_dim"])
    elif name == "transformer":
        return lambda **kwargs: Transformer(kwargs["input_dim"])
    elif name == "lightgbm":
        def _lgbm_factory(**kwargs):
            n_estimators = kwargs.get("n_estimators", 100)
            max_depth = kwargs.get("max_depth", -1)
            learning_rate = kwargs.get("learning_rate", 0.1)
            num_leaves = kwargs.get("num_leaves", 31)
            subsample = kwargs.get("subsample", 0.8)
            colsample_bytree = kwargs.get("colsample_bytree", 0.8)
            return LGBMClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, num_leaves=num_leaves,
                subsample=subsample, colsample_bytree=colsample_bytree,
                objective='binary', random_state=42, n_jobs=-1, verbose=-1,
            )
        return _lgbm_factory
    elif name == "ft_transformer":
        return lambda **kwargs: FTTransformer(kwargs["input_dim"])
    else:
        raise ValueError(f"Unknown model: {name}")


def run_simple_model(name, phase, use_text=False):
    """Run a single model using simple train/test split."""
    module_path, fn_name = SIMPLE_MODELS[name]
    display_name = NESTED_MODELS.get(name, (False, False, name))[2]
    is_dl = NESTED_MODELS.get(name, (False, False, name))[1]

    try:
        with tracker.start_run(display_name, phase=phase):
            tracker.log_model_info(display_name, _get_default_params(name), is_dl=is_dl)
            tracker.log_param("use_text", use_text)
            tracker.log_param("mode", "simple_split")

            module = importlib.import_module(module_path)
            getattr(module, fn_name)(phase, use_text=use_text)
    except Exception as e:
        print(f"  [ERROR] {name} | phase {phase}: {e}")
        traceback.print_exc()


def run_nested_model(name, phase, use_text=False, tune=False):
    """Run a single model using nested cross-validation."""
    is_tree, is_dl, display_name = NESTED_MODELS[name]
    factory_fn = _make_factory(name)

    # Check if this model-phase was already completed (resume support)
    completed_folds = tracker.get_completed_folds(display_name, phase)
    if len(completed_folds) == config.OUTER_FOLDS:
        print(f"  [SKIP] {display_name} | phase {phase} already completed ({config.OUTER_FOLDS} folds)")
        return
    elif completed_folds:
        print(f"  [RESUME] {display_name} | phase {phase} — {len(completed_folds)}/{config.OUTER_FOLDS} folds done")

    try:
        # Hyperparameter tuning for all models (sklearn: Optuna, DL: small grid)
        tune_hparams = tune
        if tune_hparams:
            print(f"  [Hyperparameter tuning enabled for {display_name}]")

        with tracker.start_run(display_name, phase=phase, nested=True):
            # Log model factory defaults as params
            tracker.log_model_info(display_name, _get_default_params(name), is_dl=is_dl)
            tracker.log_param("use_text", use_text)
            tracker.log_param("tune_hyperparams", tune_hparams)

            nested_cv_single_model(
                model_fn=factory_fn,
                phase=phase,
                model_name=display_name,
                is_tree=is_tree,
                is_dl=is_dl,
                outer_folds=config.OUTER_FOLDS,
                inner_folds=config.INNER_FOLDS,
                verbose=True,
                use_text=use_text,
                tune_hyperparams=tune_hparams,
            )
    except Exception as e:
        print(f"  [ERROR] {name} | phase {phase}: {e}")
        traceback.print_exc()

def _get_default_params(name):
    """Extract default hyperparameters from factory function for logging."""
    defaults = {
        "logistic_regression": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        "random_forest": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
        "xgboost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
        "svm": {"C": 1.0, "max_iter": 10000},
        "knn": {"n_neighbors": 5, "weights": "distance"},
        "lightgbm": {"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.1},
        "mlp": {"hidden_dim": 256, "dropout": config.DROPOUT},
        "cnn": {"conv_channels": 32, "dropout": config.DROPOUT},
        "rnn": {"hidden_dim": 64, "dropout": config.DROPOUT},
        "transformer": {"embed_dim": 64, "num_heads": 4, "num_layers": 2},
        "ft_transformer": {"embed_dim": 32, "num_heads": 4, "num_layers": 2, "hidden_dim": 64},
    }
    return defaults.get(name, {})


def main(models, phases, clear, nested, use_text=False, tune=False):
    """Run all requested models on all requested phases."""
    set_seed()

    if clear:
        clear_results()

    # ── Start MLflow experiment ──
    tracker.start_experiment(
        "SAE_Prediction",
        nested=nested,
        use_text=use_text,
        tune=tune,
        n_models=len(models),
        n_phases=len(phases),
    )
    tracker.log_config()

    for phase in phases:
        print(f"\n{'='*60}\n  PHASE {phase}\n{'='*60}")
        for name in models:
            if nested:
                print(f"\n>> Running {name} (nested CV)...")
                run_nested_model(name, phase, use_text=use_text, tune=tune)
            else:
                print(f"\n>> Running {name}...")
                run_simple_model(name, phase, use_text=use_text)

    print(f"\n{'='*60}\n  FINAL RESULTS\n{'='*60}")
    if nested:
        agg_df = aggregate_and_print()
        # Log aggregated results to MLflow
        if agg_df is not None and not agg_df.empty:
            tracker.log_aggregated_results(agg_df.to_dict(orient="records"))
    else:
        print_results_table()
        get_best_model(metric="f1")
        get_best_model(metric="roc_auc")

    from src.mlflow_tracker import MLFLOW_TRACKING_URI
    print(f"\n  [MLflow] View results: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print(f"  [MLflow] Or run: python -c \"from src.mlflow_tracker import tracker; tracker.launch_ui()\"")


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
    parser.add_argument(
        "--use-text", action="store_true",
        help="Use TF-IDF text features from trial descriptions (for supported models)"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Enable hyperparameter tuning for sklearn models (nested CV only)"
    )
    args = parser.parse_args()

    main(args.models, args.phases, args.clear, args.nested, use_text=args.use_text, tune=args.tune)
