"""
Naive Bayes Classifier Model for Credit Card Default Prediction
Probabilistic classification using Bayes theorem with feature independence assumption.
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import pickle

MODEL_NAME = "Naive Bayes"
MODEL_TYPE = "Probabilistic Classifier"
DESCRIPTION = "Probabilistic classification using Bayes theorem with Gaussian assumption"

def train_naive_bayes(X_train, y_train, X_test, y_test):
    """
    Train Naive Bayes model
    
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
    
    Returns:
    --------
    model : GaussianNB
        Trained model
    metrics : dict
        Performance metrics on test set
    """
    
    # Train model
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'mcc': matthews_corrcoef(y_test, y_pred)
    }
    
    return model, metrics

def get_class_statistics(model):
    """Get class statistics from trained model"""
    stats = {
        'theta': model.theta_,  # Mean of each feature for each class
        'var': model.var_,      # Variance of each feature for each class
        'sigma': model.sigma_   # Standard deviation of each feature for each class
    }
    return stats

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
