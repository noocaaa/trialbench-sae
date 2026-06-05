from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from src.data_loader import load_phase
from src.train_sklearn import train_sklearn_model


def run(phase, use_text=False, **kwargs):
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase, use_text=use_text)

    C = kwargs.get('C', 1.0)

    # LinearSVC is orders of magnitude faster than SVC(kernel='rbf')
    # CalibratedClassifierCV adds probability calibration with controlled CV folds
    base_svc = LinearSVC(C=C, max_iter=10000, random_state=42)
    model = CalibratedClassifierCV(base_svc, method='sigmoid', cv=3)

    # Store the base estimator C so hyperparameter tuning can access it
    model._sae_base_C = C

    model_name = "SVM+Text" if use_text else "SVM"
    train_sklearn_model(model,
                        X_train, X_test, y_train, y_test,
                        model_name=model_name, phase=phase,
                        pos_weight=pos_weight, **kwargs)
