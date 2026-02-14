import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
import os

# Load data
file_path = r'C:\Temp\Classification Models\data\default of credit card clients.csv'
df = pd.read_csv(file_path)
df = df.drop('ID', axis=1)

X = df.drop('default payment next month', axis=1)
y = df['default payment next month']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
models_dir = r'C:\Temp\Classification Models\models'
os.makedirs(models_dir, exist_ok=True)

with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved!")

# Train and save each model
models = {
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    'decision_tree': DecisionTreeClassifier(random_state=42),
    'knn': KNeighborsClassifier(n_neighbors=5),
    'naive_bayes': GaussianNB(),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'xgboost': XGBClassifier(random_state=42, verbosity=0)
}

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    model_path = os.path.join(models_dir, f'{model_name}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ {model_name} saved!")

print("\n✅ All models saved to models/ folder!")