import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import gzip

# Set page config
st.set_page_config(page_title="Credit Card Default Prediction", layout="wide")

# Title
st.title("💳 Credit Card Default Prediction")
st.markdown("Predicting customer default using 6 Machine Learning Models")

# Load models and scaler
@st.cache_resource
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl.gz',
        'Decision Tree': 'decision_tree.pkl.gz',
        'K-Nearest Neighbor': 'knn.pkl.gz',
        'Naive Bayes': 'naive_bayes.pkl.gz',
        'Random Forest': 'random_forest.pkl.gz',
        'XGBoost': 'xgboost.pkl.gz'
    }
    
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    for model_name, filename in model_files.items():
        with gzip.open(os.path.join(models_dir, filename), 'rb') as f:
            models[model_name] = pickle.load(f)
    
    with gzip.open(os.path.join(models_dir, 'scaler.pkl.gz'), 'rb') as f:
        scaler = pickle.load(f)
    
    return models, scaler

# Load model results
@st.cache_data
def load_results():
    return pd.read_csv('results/model_results.csv')

models, scaler = load_models()
results_df = load_results()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "Model Comparison", "Make Predictions"])

# ==================== PAGE 1: HOME ====================
if page == "Home":
    st.header("📊 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        st.write("""
        - **Total Samples**: 30,000 customers
        - **Features**: 24 attributes
        - **Target**: Default payment next month (Binary: 0 or 1)
        - **Default Rate**: 22.12%
        - **Train-Test Split**: 80-20
        """)
    
    with col2:
        st.subheader("Models Implemented")
        st.write("""
        1. Logistic Regression
        2. Decision Tree Classifier
        3. K-Nearest Neighbor
        4. Naive Bayes (Gaussian)
        5. Random Forest (Ensemble)
        6. XGBoost (Ensemble)
        """)
    
    st.subheader("Evaluation Metrics")
    st.write("""
    - **Accuracy**: Overall correctness
    - **AUC Score**: Ability to distinguish between classes
    - **Precision**: Of predicted defaults, how many were correct
    - **Recall**: Of actual defaults, how many were caught
    - **F1 Score**: Balance between precision and recall
    - **MCC Score**: Correlation between predicted and actual
    """)

# ==================== PAGE 2: MODEL COMPARISON ====================
elif page == "Model Comparison":
    st.header("📈 Model Performance Comparison")
    
    # Display results table
    st.subheader("Performance Metrics Table")
    
    # Format the dataframe for better display
    display_df = results_df.copy()
    for col in display_df.columns:
        if col != 'Model':
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Visualizations
    st.subheader("Model Comparisons")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Accuracy Comparison**")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(results_df['Model'], results_df['Accuracy'], color='steelblue')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0.6, 0.85])
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    
    with col2:
        st.write("**AUC Score Comparison**")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(results_df['Model'], results_df['AUC'], color='coral')
        ax.set_ylabel('AUC Score')
        ax.set_ylim([0.6, 0.8])
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    
    # Key observations
    st.subheader("📌 Key Observations")
    best_accuracy_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
    best_auc_model = results_df.loc[results_df['AUC'].idxmax(), 'Model']
    
    st.write(f"""
    - **Best Accuracy**: {best_accuracy_model} ({results_df['Accuracy'].max():.4f})
    - **Best AUC Score**: {best_auc_model} ({results_df['AUC'].max():.4f})
    - **Recommendation**: Use XGBoost or Random Forest for production (high AUC & Accuracy)
    """)

# ==================== PAGE 3: MAKE PREDICTIONS ====================
elif page == "Make Predictions":
    st.header("🔮 Make Predictions")
    
    st.write("Select a model and enter customer data to make predictions")
    
    # Model selection
    selected_model = st.selectbox("Select a Model:", list(models.keys()))
    
    # Display model performance
    model_row = results_df[results_df['Model'] == selected_model].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{model_row['Accuracy']:.4f}")
    with col2:
        st.metric("AUC", f"{model_row['AUC']:.4f}")
    with col3:
        st.metric("Precision", f"{model_row['Precision']:.4f}")
    with col4:
        st.metric("Recall", f"{model_row['Recall']:.4f}")
    
    st.divider()
    
    # Input section
    st.subheader("Enter Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        limit_bal = st.number_input("Credit Limit (NT$)", min_value=10000, max_value=1000000, value=50000, step=10000)
        sex = st.selectbox("Sex", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        education = st.selectbox("Education", [1, 2, 3, 4], format_func=lambda x: {1: "Graduate school", 2: "University", 3: "High school", 4: "Others"}[x])
        marriage = st.selectbox("Marriage Status", [1, 2, 3], format_func=lambda x: {1: "Married", 2: "Single", 3: "Divorced"}[x])
    
    with col2:
        age = st.slider("Age", min_value=21, max_value=79, value=35)
        #pay_0 = st.selectbox("Recent Payment Status", [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], value=0)
        pay_0 = st.selectbox("Recent Payment Status", [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=1)
        bill_amt1 = st.number_input("Recent Bill Amount (NT$)", min_value=0, max_value=900000, value=50000, step=1000)
        pay_amt1 = st.number_input("Recent Payment Amount (NT$)", min_value=0, max_value=500000, value=10000, step=1000)
    
    with col3:
        #pay_2 = st.selectbox("Payment Status (2 months ago)", [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], value=0)
        pay_2 = st.selectbox("Payment Status (2 months ago)", [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=1)
        bill_amt2 = st.number_input("Bill Amount (2 months ago) (NT$)", min_value=0, max_value=900000, value=45000, step=1000)
        pay_amt2 = st.number_input("Payment Amount (2 months ago) (NT$)", min_value=0, max_value=500000, value=9000, step=1000)
    
    # Create input array with all 24 features (simplified for demo)
    # In production, you'd need all 24 features
    input_data = np.array([[
        limit_bal, sex, education, marriage, age,
        pay_0, pay_2, 0, 0, 0, 0,  # PAY_3 to PAY_6
        bill_amt1, bill_amt2, 0, 0, 0, 0,  # BILL_AMT3 to BILL_AMT6
        pay_amt1, pay_amt2, 0, 0, 0, 0  # PAY_AMT3 to PAY_AMT6
    ]])
    
    # Make prediction
    if st.button("🔮 Make Prediction", key="predict_button"):
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Get prediction and probability
        model = models[selected_model]
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        st.divider()
        st.subheader("Prediction Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ HIGH RISK: Customer likely to DEFAULT")
            else:
                st.success("✅ LOW RISK: Customer likely to PAY")
        
        with col2:
            st.write(f"**Confidence**: {max(probability) * 100:.2f}%")
            st.write(f"- Probability of Default: {probability[1] * 100:.2f}%")
            st.write(f"- Probability of Payment: {probability[0] * 100:.2f}%")

st.divider()
st.write("---")
st.write("📝 **Created by**: Priya S | **Course**: Machine Learning Assignment 2")