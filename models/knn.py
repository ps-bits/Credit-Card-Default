"""
K-Nearest Neighbor (kNN) Model for Credit Card Default Prediction
Instance-based learning using distance to neighboring samples.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import pickle

MODEL_NAME = "K-Nearest Neighbor"
MODEL_TYPE = "Instance-based Classifier"
DESCRIPTION = "Classification based on nearest neighbor distances in feature space"

def train_knn(X_train, y_train, X_test, y_test, n_neighbors=5, scaler=None):
    """
    Train K-Nearest Neighbor model
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    X_test : array-like
        Testing features
    y_test : array-like
        Testing target
    n_neighbors : int, default=5
        Number of neighbors to consider
    scaler : StandardScaler, optional
        Pre-fitted scaler for feature normalization
    
    Returns:
    --------
    model : KNeighborsClassifier
        Trained model
    scaler : StandardScaler
        Fitted scaler
    metrics : dict
        Performance metrics on test set
    """
    
    # Scale features (important for kNN)
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'mcc': matthews_corrcoef(y_test, y_pred)
    }
    
    return model, scaler, metrics

def save_model(model, filepath):
    """Save model to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load model from file"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

if __name__ == "__main__":
    print(f"Module: {MODEL_NAME}")
    print(f"Type: {MODEL_TYPE}")
    print(f"Description: {DESCRIPTION}")
