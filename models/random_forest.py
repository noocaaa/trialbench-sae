from sklearn.ensemble import RandomForestClassifier
from src.data_loader import load_phase
from src.train_sklearn import train_sklearn_model


def run(phase, **kwargs):
    # Tree models use ordinal encoding + no standardization
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase, for_tree=True)

    n_estimators = kwargs.get('n_estimators', 100)
    max_depth = kwargs.get('max_depth', None)
    min_samples_split = kwargs.get('min_samples_split', 2)
    min_samples_leaf = kwargs.get('min_samples_leaf', 1)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )

    train_sklearn_model(model,
                        X_train, X_test, y_train, y_test,
                        model_name="RandomForest", phase=phase,
                        pos_weight=pos_weight, **kwargs)
