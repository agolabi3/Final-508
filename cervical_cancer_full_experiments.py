"""
CIS 508 Final Project – Cervical Cancer Risk Prediction
Full Experiment Script (20+ ML Models Across 8 Families)
-------------------------------------------------------
This script is designed for LOCAL / GITHUB execution, not Colab.

✔ No Colab-specific commands
✔ No `!pip install`
✔ Uses argparse for CLI arguments
✔ Logs all runs to Databricks MLflow (if DATABRICKS_* env vars are set)
✔ Evaluates: Accuracy, Precision, Recall, F1
✔ Picks best model by Recall (most important for healthcare screening)

Usage:
    python cervical_cancer_full_experiments.py --csv_path path/to/risk_factors_cervical_cancer.csv

Optional:
    --experiment_name "/Users/your_email@asu.edu/CervicalCancer_FinalProject"
"""

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

import mlflow
import mlflow.sklearn
import joblib


def evaluate_model(name: str, y_true, y_pred):
    """Compute and print standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n📊 {name} Results")
    print("---------------------------")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}


def main():
    parser = argparse.ArgumentParser(
        description="Run 20+ ML experiments for cervical cancer risk prediction and log to Databricks MLflow."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to risk_factors_cervical_cancer.csv (from the Kaggle Cervical Cancer Risk dataset).",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="/Users/agolabi3@asu.edu/CervicalCancer_FinalProject",
        help="Databricks MLflow experiment path.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load and prepare dataset
    # ------------------------------------------------------------------
    print(f"📂 Loading dataset from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    # Replace '?' with NaN and convert numeric where possible
    df.replace("?", np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors="ignore")

    # Drop rows missing the target
    if "Biopsy" not in df.columns:
        raise KeyError("Expected target column 'Biopsy' not found in dataset.")
    df.dropna(subset=["Biopsy"], inplace=True)

    X = df.drop(columns=["Biopsy"])
    y = df["Biopsy"]

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standard scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("✅ Data cleaned, imputed, scaled, and split.")
    print("   Train shape:", X_train.shape, " Test shape:", X_test.shape)

    # ------------------------------------------------------------------
    # 2. Configure MLflow with Databricks
    # ------------------------------------------------------------------
    print("\n🔗 Configuring MLflow to use Databricks tracking URI...")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(args.experiment_name)

    print("   Tracking URI:", mlflow.get_tracking_uri())
    print("   Experiment  :", args.experiment_name)

    # ------------------------------------------------------------------
    # 3. Hyperparameter grids for 8 model families (>= 2 configs each)
    # ------------------------------------------------------------------
    log_param_grid = [
        {"C": 0.1, "penalty": "l2", "solver": "lbfgs"},
        {"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
        {"C": 10.0, "penalty": "l2", "solver": "lbfgs"},
    ]

    svm_param_grid = [
        {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
        {"C": 0.1, "kernel": "rbf", "gamma": "scale"},
    ]

    dt_param_grid = [
        {"max_depth": 2, "min_samples_split": 2},
        {"max_depth": 3, "min_samples_split": 4},
    ]

    rf_param_grid = [
        {"n_estimators": 50, "max_depth": 2},
        {"n_estimators": 100, "max_depth": 3},
        {"n_estimators": 200, "max_depth": None},
    ]

    knn_param_grid = [
        {"n_neighbors": 3, "weights": "uniform"},
        {"n_neighbors": 5, "weights": "distance"},
    ]

    nb_param_grid = [
        {"var_smoothing": 1e-9},
        {"var_smoothing": 1e-7},
    ]

    gb_param_grid = [
        {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 2},
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3},
    ]

    ann_param_grid = [
        {"hidden_layer_sizes": (16,), "alpha": 0.0001},
        {"hidden_layer_sizes": (32, 16), "alpha": 0.0005},
        {"hidden_layer_sizes": (64, 32, 16), "alpha": 0.001},
    ]

    total_runs = (
        len(log_param_grid)
        + len(svm_param_grid)
        + len(dt_param_grid)
        + len(rf_param_grid)
        + len(knn_param_grid)
        + len(nb_param_grid)
        + len(gb_param_grid)
        + len(ann_param_grid)
    )

    print(f"\n🚀 Will run {total_runs} experiments across 8 model families.\n")

    # ------------------------------------------------------------------
    # 4. Track best model by Recall
    # ------------------------------------------------------------------
    best_model = None
    best_family = None
    best_run_name = None
    best_params = None
    best_metrics = {"recall": -1.0}

    def update_best(model_family, run_name, params, metrics, model_obj):
        nonlocal best_model, best_family, best_run_name, best_params, best_metrics
        if metrics["recall"] > best_metrics["recall"]:
            best_metrics = metrics
            best_model = model_obj
            best_family = model_family
            best_run_name = run_name
            best_params = params

    # ------------------------------------------------------------------
    # 5. Run all experiments and log them to MLflow
    # ------------------------------------------------------------------

    # Logistic Regression
    for i, params in enumerate(log_param_grid, start=1):
        run_name = f"LOGREG_C{params['C']}_{params['solver']}"
        print(f"\n=== Logistic Regression {i}/{len(log_param_grid)}: {run_name} ===")

        with mlflow.start_run(run_name=run_name):
            model = LogisticRegression(
                C=params["C"],
                penalty=params["penalty"],
                solver=params["solver"],
                max_iter=1000,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate_model(run_name, y_test, preds)

            mlflow.log_param("model_family", "LogisticRegression")
            mlflow.log_param("C", params["C"])
            mlflow.log_param("penalty", params["penalty"])
            mlflow.log_param("solver", params["solver"])
            mlflow.log_param("max_iter", 1000)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            update_best("LogisticRegression", run_name, params, metrics, model)

    # SVM
    for i, params in enumerate(svm_param_grid, start=1):
        run_name = f"SVM_C{params['C']}_{params['kernel']}"
        print(f"\n=== SVM {i}/{len(svm_param_grid)}: {run_name} ===")

        with mlflow.start_run(run_name=run_name):
            model = SVC(
                C=params["C"],
                kernel=params["kernel"],
                gamma=params["gamma"],
                probability=False,
                random_state=42,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate_model(run_name, y_test, preds)

            mlflow.log_param("model_family", "SVM")
            mlflow.log_param("C", params["C"])
            mlflow.log_param("kernel", params["kernel"])
            mlflow.log_param("gamma", params["gamma"])
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            update_best("SVM", run_name, params, metrics, model)

    # Decision Tree
    for i, params in enumerate(dt_param_grid, start=1):
        run_name = f"DT_depth{params['max_depth']}_minsplit{params['min_samples_split']}"
        print(f"\n=== Decision Tree {i}/{len(dt_param_grid)}: {run_name} ===")

        with mlflow.start_run(run_name=run_name):
            model = DecisionTreeClassifier(
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                random_state=42,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate_model(run_name, y_test, preds)

            mlflow.log_param("model_family", "DecisionTree")
            mlflow.log_param("max_depth", params["max_depth"])
            mlflow.log_param("min_samples_split", params["min_samples_split"])
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            update_best("DecisionTree", run_name, params, metrics, model)

    # Random Forest
    for i, params in enumerate(rf_param_grid, start=1):
        run_name = f"RF_n{params['n_estimators']}_depth{params['max_depth']}"
        print(f"\n=== Random Forest {i}/{len(rf_param_grid)}: {run_name} ===")

        with mlflow.start_run(run_name=run_name):
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate_model(run_name, y_test, preds)

            mlflow.log_param("model_family", "RandomForest")
            mlflow.log_param("n_estimators", params["n_estimators"])
            mlflow.log_param("max_depth", params["max_depth"])
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            update_best("RandomForest", run_name, params, metrics, model)

    # KNN
    for i, params in enumerate(knn_param_grid, start=1):
        run_name = f"KNN_k{params['n_neighbors']}_{params['weights']}"
        print(f"\n=== KNN {i}/{len(knn_param_grid)}: {run_name} ===")

        with mlflow.start_run(run_name=run_name):
            model = KNeighborsClassifier(
                n_neighbors=params["n_neighbors"],
                weights=params["weights"],
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate_model(run_name, y_test, preds)

            mlflow.log_param("model_family", "KNN")
            mlflow.log_param("n_neighbors", params["n_neighbors"])
            mlflow.log_param("weights", params["weights"])
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            update_best("KNN", run_name, params, metrics, model)

    # Naive Bayes
    for i, params in enumerate(nb_param_grid, start=1):
        run_name = f"NB_vsmooth{params['var_smoothing']}"
        print(f"\n=== Naive Bayes {i}/{len(nb_param_grid)}: {run_name} ===")

        with mlflow.start_run(run_name=run_name):
            model = GaussianNB(var_smoothing=params["var_smoothing"])
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate_model(run_name, y_test, preds)

            mlflow.log_param("model_family", "NaiveBayes")
            mlflow.log_param("var_smoothing", params["var_smoothing"])
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            update_best("NaiveBayes", run_name, params, metrics, model)

    # Gradient Boosting
    for i, params in enumerate(gb_param_grid, start=1):
        run_name = f"GB_n{params['n_estimators']}_lr{params['learning_rate']}_depth{params['max_depth']}"
        print(f"\n=== Gradient Boosting {i}/{len(gb_param_grid)}: {run_name} ===")

        with mlflow.start_run(run_name=run_name):
            model = GradientBoostingClassifier(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                random_state=42,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate_model(run_name, y_test, preds)

            mlflow.log_param("model_family", "GradientBoosting")
            mlflow.log_param("n_estimators", params["n_estimators"])
            mlflow.log_param("learning_rate", params["learning_rate"])
            mlflow.log_param("max_depth", params["max_depth"])
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            update_best("GradientBoosting", run_name, params, metrics, model)

    # ANN / MLP
    for i, params in enumerate(ann_param_grid, start=1):
        run_name = f"ANN_layers{params['hidden_layer_sizes']}_alpha{params['alpha']}"
        print(f"\n=== ANN {i}/{len(ann_param_grid)}: {run_name} ===")

        with mlflow.start_run(run_name=run_name):
            model = MLPClassifier(
                hidden_layer_sizes=params["hidden_layer_sizes"],
                alpha=params["alpha"],
                max_iter=1000,
                random_state=42,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate_model(run_name, y_test, preds)

            mlflow.log_param("model_family", "ANN_MLP")
            mlflow.log_param("hidden_layer_sizes", params["hidden_layer_sizes"])
            mlflow.log_param("alpha", params["alpha"])
            mlflow.log_param("max_iter", 1000)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            update_best("ANN", run_name, params, metrics, model)

    # ------------------------------------------------------------------
    # 6. Summary and save best model
    # ------------------------------------------------------------------
    print("\n🏆 BEST RUN (by Recall):")
    print("---------------------------")
    print("Model Family:", best_family)
    print("Run Name    :", best_run_name)
    print("Params      :", best_params)
    print("Metrics     :", best_metrics)

    print(
        """
✅ NOTE:
In this healthcare screening context, RECALL (sensitivity) is the most important metric.
We prefer the model that misses the fewest true cancer cases (minimizing false negatives),
even if that means a slight trade-off in precision or accuracy.
"""
    )

    # Save the best model locally
    out_path = "best_cervical_model.pkl"
    joblib.dump(best_model, out_path)
    print(f"💾 Saved best model to: {out_path}")


if __name__ == "__main__":
    main()
