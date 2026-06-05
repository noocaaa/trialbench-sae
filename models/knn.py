from sklearn.neighbors import KNeighborsClassifier
from src.data_loader import load_phase
from src.train_sklearn import train_sklearn_model


def run(phase, **kwargs):
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase)

    k = kwargs.get('k', 5)
    weights = kwargs.get('weights', 'distance')

    model = KNeighborsClassifier(
        n_neighbors=k,
        weights=weights,
        n_jobs=-1
    )

    train_sklearn_model(model,
                        X_train, X_test, y_train, y_test,
                        model_name="KNN", phase=phase,
                        pos_weight=pos_weight, **kwargs)
