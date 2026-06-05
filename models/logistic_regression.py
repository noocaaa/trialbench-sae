from sklearn.linear_model import LogisticRegression
from src.data_loader import load_phase
from src.train_sklearn import train_sklearn_model


def run(phase, use_text=False, **kwargs):
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase, use_text=use_text)

    # Get hyperparameters from kwargs or use defaults
    C = kwargs.get('C', 1.0)
    max_iter = kwargs.get('max_iter', 1000)
    solver = kwargs.get('solver', 'lbfgs')

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        random_state=42
    )

    model_name = "LogisticRegression+Text" if use_text else "LogisticRegression"
    train_sklearn_model(model,
                        X_train, X_test, y_train, y_test,
                        model_name=model_name, phase=phase,
                        pos_weight=pos_weight, **kwargs)
