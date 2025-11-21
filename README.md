# Final-508
**Model Development & Experimentation**

This project implements a complete, reproducible machine-learning workflow for predicting cervical cancer biopsy outcomes using the Cervical Cancer Risk Factors dataset (Kaggle). All model training, hyperparameter exploration, and metric logging are performed using Databricks MLflow, ensuring end-to-end experiment tracking and version control.

**Overview of the Experiment Script**

The core experimentation pipeline is implemented in:

cervical_cancer_full_experiments.py

This script runs 20+ machine learning experiments across 8 different model families, evaluating each model on a consistent train/test split. The goal is to identify which algorithm and configuration provides the highest recall (sensitivity), which is the most important metric for medical screening applications.

**Key Features**

Fully automated experiment runner

Covers 8 classification model families:

Logistic Regression

Support Vector Machine (SVM)

Decision Tree

Random Forest

K-Nearest Neighbors (KNN)

Naive Bayes

Gradient Boosting

Artificial Neural Network (ANN / MLP)

Runs multiple hyperparameter configurations for each model

Logs accuracy, precision, recall, and F1-score to MLflow

Uses Databricks MLflow for experiment tracking

Selects the best model based on recall to minimize false negatives

Saves best model as:
best_cervical_model.pkl 
