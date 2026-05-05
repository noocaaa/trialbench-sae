import argparse
import importlib

from src.utils import clear_results, print_results_table, get_best_model

PHASES = ["1", "2", "3", "4"]

MODELS = {
    # Traditional ML (Fahad)
    "logistic_regression": ("models.logistic_regression", "run"),
    "random_forest":       ("models.random_forest",       "run"),
    "svm":             ("models.svm",             "run"),
    "knn":             ("models.knn",             "run"),

    # Deep Learning (Noelia)
    "mlp":                 ("models.mlp",                 "run"),
    "cnn":                 ("models.cnn",                 "run"),
    "rnn":                 ("models.rnn",                 "run"),
    "transformer":         ("models.transformer",         "run"),
}


def run_model(name, phase):
    module_path, fn_name = MODELS[name]

    try:
        module = importlib.import_module(module_path)
        getattr(module, fn_name)(phase)
    except Exception as e:
        print(f"  [ERROR] {name} | phase {phase}: {e}")

def main(models, phases, clear):
    if clear:
        clear_results()

    for phase in phases:
        print(f"\n{'='*40}\n  Phase {phase}\n{'='*40}")
        for name in models:
            print(f"\n>> Running {name}...")
            run_model(name, phase)

    print(f"\n{'='*40}\n  Final Results\n{'='*40}")
    print_results_table()
    get_best_model(metric="f1")
    get_best_model(metric="roc_auc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all models for SAE prediction.")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()), default=list(MODELS.keys()),
                        help="Models to run (default: all)")
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=PHASES,
                        help="Trial phases to evaluate (default: all)")
    parser.add_argument("--clear", action="store_true",
                        help="Clear previous results before running")
    args = parser.parse_args()

    main(args.models, args.phases, args.clear)