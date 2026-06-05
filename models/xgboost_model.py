from xgboost import XGBClassifier
from src.data_loader import load_phase
from src.train_sklearn import train_sklearn_model


def run(phase, **kwargs):
    # Tree models use ordinal encoding + no standardization
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase, for_tree=True)

    n_estimators = kwargs.get('n_estimators', 100)
    max_depth = kwargs.get('max_depth', 6)
    learning_rate = kwargs.get('learning_rate', 0.1)
    subsample = kwargs.get('subsample', 0.8)
    colsample_bytree = kwargs.get('colsample_bytree', 0.8)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
    )

    train_sklearn_model(model,
                        X_train, X_test, y_train, y_test,
                        model_name="XGBoost", phase=phase,
                        pos_weight=pos_weight, **kwargs)
