"""
XGBoost Ensemble Model for Credit Card Default Prediction
Gradient boosting with optimized tree learning for superior performance.
"""

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import pickle

MODEL_NAME = "XGBoost"
MODEL_TYPE = "Gradient Boosting Ensemble"
DESCRIPTION = "Gradient boosting with optimized tree learning for superior discrimination ability"

def train_xgboost(X_train, y_train, X_test, y_test, scaler=None):
    """
    Train XGBoost model
    
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
    scaler : StandardScaler, optional
        Pre-fitted scaler for feature normalization
    
    Returns:
    --------
    model : XGBClassifier
        Trained model
    scaler : StandardScaler
        Fitted scaler
    metrics : dict
        Performance metrics on test set
    """
    
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = XGBClassifier(random_state=42, verbosity=0)
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

def get_feature_importance(model, feature_names):
    """
    Get feature importance from trained XGBoost
    
    Parameters:
    -----------
    model : XGBClassifier
        Trained model
    feature_names : list
        Names of features
    
    Returns:
    --------
    dict : Feature importance scores sorted in descending order
    """
    importances = model.feature_importances_
    feature_importance_dict = dict(zip(feature_names, importances))
    return sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

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
