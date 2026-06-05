from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from src.data_loader import load_phase
from src.train_sklearn import train_sklearn_model


def run(phase, **kwargs):
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase)

    C = kwargs.get('C', 1.0)

    # LinearSVC is orders of magnitude faster than SVC(kernel='rbf')
    # CalibratedClassifierCV adds probability calibration with controlled CV folds
    base_svc = LinearSVC(C=C, max_iter=10000, random_state=42)
    model = CalibratedClassifierCV(base_svc, method='sigmoid', cv=3)

    train_sklearn_model(model,
                        X_train, X_test, y_train, y_test,
                        model_name="SVM", phase=phase,
                        pos_weight=pos_weight, **kwargs)
