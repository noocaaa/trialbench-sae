from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_phase
from src.train_sklearn import train_sklearn_model


def run(phase, **kwargs):
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase)
    
    C = kwargs.get('C', 1.0)
    kernel = kwargs.get('kernel', 'rbf')
    gamma = kwargs.get('gamma', 'scale')
    
    # Scale features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=True,  # Need this for predict_proba
        random_state=42
    )
    
    train_sklearn_model(model,
                        X_train_scaled, X_test_scaled, y_train, y_test,
                        model_name="SVM", phase=phase, **kwargs)