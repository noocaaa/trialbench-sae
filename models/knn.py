from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_phase
from src.train_sklearn import train_sklearn_model


def run(phase, **kwargs):
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase)
    
    k = kwargs.get('k', 5)
    weights = kwargs.get('weights', 'distance')
    
    # Scale features for KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = KNeighborsClassifier(
        n_neighbors=k,
        weights=weights,
        n_jobs=-1
    )
    
    train_sklearn_model(model,
                        X_train_scaled, X_test_scaled, y_train, y_test,
                        model_name=f"KNN_k{k}", phase=phase, **kwargs)