from sklearn.linear_model import LogisticRegression
from src.data_loader import load_phase
from src.train_sklearn import train_sklearn_model


def run(phase, **kwargs):
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase)
    
    # Get hyperparameters from kwargs or use defaults
    C = kwargs.get('C', 1.0)
    max_iter = kwargs.get('max_iter', 1000)
    solver = kwargs.get('solver', 'lbfgs')
    class_weight = kwargs.get('class_weight', 'balanced')
    
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        class_weight=class_weight,
        random_state=42
    )
    
    train_sklearn_model(model,
                        X_train, X_test, y_train, y_test,
                        model_name="LogisticRegression", phase=phase, **kwargs)