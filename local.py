import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)

from trialbench.function import load_data


def load_phase(phase, data_dir):
    """
    Load a single phase's data from CSV files.
    Handles string columns (IDs, categoricals) automatically.
    """
    phase_dir = os.path.join(data_dir, f'Phase{phase}')

    # Load raw CSVs (let pandas infer types)
    train_x = pd.read_csv(os.path.join(phase_dir, 'train_x.csv'))
    train_y = pd.read_csv(os.path.join(phase_dir, 'train_y.csv'))
    test_x = pd.read_csv(os.path.join(phase_dir, 'test_x.csv'))
    test_y = pd.read_csv(os.path.join(phase_dir, 'test_y.csv'))

    print(f"\n--- Phase {phase} raw shapes ---")
    print(f"train_x: {train_x.shape}")
    print(f"train_y: {train_y.shape}")
    print(f"test_x:  {test_x.shape}")
    print(f"test_y:  {test_y.shape}")

    # ---- Handle X (features) ----
    # Drop obvious ID columns (names containing 'id', 'nct', 'trial', 'study')
    id_cols = [c for c in train_x.columns
               if any(k in c.lower() for k in ['id', 'nct', 'trial', 'study', 'name'])]
    if id_cols:
        print(f"Dropping ID columns: {id_cols}")
        train_x = train_x.drop(columns=id_cols)
        test_x = test_x.drop(columns=id_cols)

    # Separate numeric and categorical columns
    numeric_cols = train_x.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train_x.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numeric cols: {len(numeric_cols)}, Categorical cols: {len(cat_cols)}")
    if cat_cols:
        print(f"Categorical columns: {cat_cols}")

    # For remaining object columns, try to coerce to numeric first
    for col in cat_cols[:]:
        train_converted = pd.to_numeric(train_x[col], errors='coerce')
        test_converted = pd.to_numeric(test_x[col], errors='coerce')
        # If most values converted successfully, keep as numeric
        if train_converted.notna().sum() / len(train_x) > 0.5:
            train_x[col] = train_converted
            test_x[col] = test_converted
            numeric_cols.append(col)
            cat_cols.remove(col)

    # One-hot encode any remaining true categorical columns
    if cat_cols:
        # Combine train+test for consistent encoding
        combined = pd.concat([train_x[cat_cols], test_x[cat_cols]], axis=0)
        combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

        n_train = len(train_x)
        train_cat = combined_encoded.iloc[:n_train].reset_index(drop=True)
        test_cat = combined_encoded.iloc[n_train:].reset_index(drop=True)

        # Drop original categorical cols and join encoded ones
        train_x = train_x.drop(columns=cat_cols).reset_index(drop=True)
        test_x = test_x.drop(columns=cat_cols).reset_index(drop=True)
        train_x = pd.concat([train_x, train_cat], axis=1)
        test_x = pd.concat([test_x, test_cat], axis=1)

    # Align columns (in case test is missing some dummy columns)
    train_x, test_x = train_x.align(test_x, join='left', axis=1, fill_value=0)

    # Convert to numpy arrays
    X_train = train_x.values.astype(np.float32)
    X_test = test_x.values.astype(np.float32)

    # Fill any NaN with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Handle y (labels) ----
    if train_y.shape[1] == 1:
        y_train = train_y.values.flatten().astype(int)
        y_test = test_y.values.flatten().astype(int)
    else:
        # Multi-column labels: take last column as binary indicator
        y_train = train_y.iloc[:, -1].values.astype(int)
        y_test = test_y.iloc[:, -1].values.astype(int)

    return X_train, y_train, X_test, y_test

def get_models():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight='balanced'
        ),
        'SVM': SVC(
            kernel='rbf', C=1.0, gamma='scale', probability=True,
            random_state=42, class_weight='balanced'
        ),
        'KNN (K=5)': KNeighborsClassifier(
            n_neighbors=5, weights='distance', metric='euclidean', n_jobs=-1
        )
    }

def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test, scaler=None):
    print(f"\n{'-'*50}")
    print(f"Training: {model_name}")
    print(f"{'-'*50}")

    if scaler is not None and model_name != 'Random Forest':
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
        'PR-AUC': average_precision_score(y_test, y_prob) if y_prob is not None else np.nan,
    }

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No SAE', 'SAE']))
    print(f"\nDetailed Metrics:")
    for key, val in metrics.items():
        if key != 'Model':
            print(f"  {key:<15}: {val:.4f}")

    return model, metrics, y_prob

def run_experiment(phase=1, data_dir='./data/serious-adverse-event-forecasting'):
    X_train, y_train, X_test, y_test = load_phase(phase, data_dir)
    scaler = StandardScaler()
    models = get_models()

    all_results = []
    all_probs = {}

    for name, model in models.items():
        _, metrics, y_prob = train_and_evaluate(
            model, name, X_train, y_train, X_test, y_test, scaler=scaler
        )
        all_results.append(metrics)
        all_probs[name] = y_prob

    results_df = pd.DataFrame(all_results)
    print(f"\n{'='*60}")
    print(f"SUMMARY - Phase {phase}")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))

    results_df.to_csv(f'sae_results_phase_{phase}.csv', index=False)
    print(f"\nResults saved to: sae_results_phase_{phase}.csv")

    return results_df

if __name__ == '__main__':
    # UPDATE THIS to your actual folder path
    DATA_DIR = r'D:.\data\serious-adverse-event-forecasting'

    # Run the experiment for a specific phase or 'All'
    results = run_experiment(1, DATA_DIR)

    # test the trialbench loader to see if it can find the labels
    # train_loader, valid_loader, test_loader, num_classes, tabular_input_dim = load_data('serious_adverse_rate_yn', '1', data_format='df')
    # print(f"train_loader: {len(train_loader)}")