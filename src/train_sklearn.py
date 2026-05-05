import os
import json
import numpy as np
from src.evaluate import evaluate

def train_sklearn_model(model, X_train, X_test, y_train, y_test, model_name, phase, **kwargs):
    """
    Training function for sklearn models.
    
    Parameters
    ----------
    model      : sklearn model — any model with fit() and predict_proba() methods
    X_train    : np.ndarray — preprocessed training features
    X_test     : np.ndarray — preprocessed test features
    y_train    : np.ndarray — training labels (0/1)
    y_test     : np.ndarray — test labels (0/1)
    model_name : str        — e.g. "LogisticRegression", "RandomForest"
    phase      : str        — "1", "2", "3", "4"
    kwargs     : dict       — additional arguments (passed through)
    """    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Get probabilities (if available)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # For models without predict_proba (like SVM without probability=True)
        y_prob = y_pred.astype(float)
    
    # Evaluate using your existing evaluation function
    evaluate(y_test, y_pred, y_prob, model_name=model_name, phase=phase)
    
    # Save model info (optional)
    save_model_info(model, model_name, phase)
    
    return model


def save_model_info(model, model_name, phase):
    """Save model hyperparameters and info."""
    os.makedirs("results", exist_ok=True)
    
    info = {
        'model_type': model_name,
        'phase': phase,
        'model_params': model.get_params()
    }
    
    # Add model-specific info
    if hasattr(model, 'n_features_in_'):
        info['n_features_in'] = model.n_features_in_
    
    with open(f"results/{model_name}_{phase}_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)