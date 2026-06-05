from sklearn.neighbors import KNeighborsClassifier
from src.data_loader import load_phase
from src.train_sklearn import train_sklearn_model


def run(phase, use_text=False, **kwargs):
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase, use_text=use_text)

    n_neighbors = kwargs.get('n_neighbors', 5)
    weights = kwargs.get('weights', 'distance')

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        n_jobs=-1
    )

    model_name = "KNN+Text" if use_text else "KNN"
    train_sklearn_model(model,
                        X_train, X_test, y_train, y_test,
                        model_name=model_name, phase=phase,
                        pos_weight=pos_weight, **kwargs)
