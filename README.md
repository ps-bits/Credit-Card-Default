
# Credit-Card-Default
Credit Card Default Prediction using 6 ML Models

PROBLEM STATEMENT
-----------------
This project implements machine learning classification models to predict whether a credit card customer will default on their payment in the next month. This is a critical problem for financial institutions to manage credit risk effectively.

DATASET DESCRIPTION
--------------------
Source: UCI Machine Learning Repository - Default of Credit Card Clients
Total Samples: 30,000 customers
Features: 24 attributes including:
• Demographics: Sex, Education, Marriage Status, Age
• Payment History: PAY_0 to PAY_6 (6 months of payment status)
• Bill Amounts: BILL_AMT1 to BILL_AMT6 (6 months of billing amounts)
• Payment Amounts: PAY_AMT1 to PAY_AMT6 (6 months of payment amounts)
• Credit Limit: LIMIT_BAL
Target Variable: Default payment next month (Binary: 0=No default, 1=Default)
Class Distribution: 77.88% No Default, 22.12% Default
Train-Test Split: 80% training, 20% testing


MODELS USED
-----------
ML Model Name 	          Accuracy 	AUC 	Precision 	Recall 	F1 	MCC 
Logistic Regression	      0.8100 	0.7270	0.6927	0.2369	0.3530	0.3259
Decision Tree	            0.7250 	0.6108 	0.3802 	0.4075	0.3934	0.2161
kNN 	                    0.7950 	0.7078 	0.5487 	0.3564 	0.4321	0.3247
Naive Bayes 	            0.7070 	0.7371 	0.3967 	0.6504 	0.4928 	0.3218
Random Forest (Ensemble)	0.8160 	0.7570	0.6384	0.3671	0.4662	0.3850
XGBoost (Ensemble)	      0.8148 	0.7750	0.6347	0.3625	0.4615	0.3801


OBSERVATION ON THE PERFORMANCE
-------------------------------
Logistic Regression: 
Good accuracy (81%) and best precision (0.693), indicating few false positives when predicting defaults. However, low recall (0.237) means it misses many actual defaulters. This conservative model is suitable when minimizing false alarms is important.

Decision Tree:
Worst performer with 72.5% accuracy and lowest AUC (0.611). Likely suffering from overfitting or underfitting. The model struggles to capture the underlying patterns in the credit card data. Not recommended for production use without significant hyperparameter tuning.

K-Nearest Neighbor (kNN):
Moderate performance (79.5% accuracy). Provides a reasonable balance between precision (0.549) and recall (0.356). Slower prediction time due to distance calculations, making it less suitable for real-time systems. Performance depends heavily on feature scaling.

Naive Bayes:
High AUC (0.737) and excellent recall (0.650) - catches most defaulters. However, lowest accuracy (70.7%) due to many false positives. Useful when identifying all potential defaults is critical, even if some are false alarms. Good probabilistic approach to the problem.

Random Forest (Ensemble):
Best accuracy (81.6%) with excellent AUC (0.757). Good balance across all metrics. Strong ensemble method that handles feature interactions well and reduces overfitting. Highly recommended for production use due to robust performance.

XGBoost (Ensemble):
Highest AUC score (0.775) - best at distinguishing between defaulters and non-defaulters. Achieves 81.48% accuracy with strong gradient boosting approach. Most suitable for production deployment due to superior discrimination ability and ability to handle complex patterns in imbalanced data.


