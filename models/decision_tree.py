"""
Decision Tree Classifier Model for Credit Card Default Prediction
Non-linear classification using hierarchical decision rules.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import pickle

MODEL_NAME = "Decision Tree Classifier"
MODEL_TYPE = "Tree-based Classifier"
DESCRIPTION = "Non-linear classification using decision trees with interpretable rules"

def train_decision_tree(X_train, y_train, X_test, y_test):
    """
    Train Decision Tree model
    
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
    model : DecisionTreeClassifier
        Trained model
    metrics : dict
        Performance metrics on test set
    """
    
    # Train model
    model = DecisionTreeClassifier(random_state=42)
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

def get_feature_importance(model, feature_names):
    """Get feature importance from trained tree"""
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
