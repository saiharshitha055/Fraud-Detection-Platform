import os
import pandas as pd
import numpy as np
import logging
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# Import our feature pipeline functions
from ml.features.build_features import load_raw_data, engineer_features, time_based_train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_models():
    # 1. Load and process data using our reproducible pipeline
    raw_path = "ml/data/raw/PS_20174392719_1491204439457_log.csv"
    df = load_raw_data(raw_path)
    processed_df = engineer_features(df)
    
    X_train, X_test, y_train, y_test = time_based_train_test_split(processed_df)
    
    # Set MLflow experiment name
    mlflow.set_experiment("fraud_detection_experiment")
    
    # ---------------------------------------------------------
    # Model 1: Baseline Logistic Regression
    # ---------------------------------------------------------
    logging.info("Training baseline Logistic Regression model...")
    with mlflow.start_run(run_name="logistic_regression_baseline"):
        lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lr_model.fit(X_train, y_train)
        
        preds = lr_model.predict(X_test)
        probs = lr_model.predict_proba(X_test)[:, 1]
        
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc_auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        
        # Log metrics to MLflow
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        
        mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
        logging.info(f"Logistic Regression Metrics -> Precision: {precision:.4f}, Recall: {recall:.4f}, ROC-AUC: {roc_auc:.4f}")

    # ---------------------------------------------------------
    # Model 2: Advanced XGBoost Classifier
    # ---------------------------------------------------------
    logging.info("Training advanced XGBoost Classifier...")
    
    # Calculate scale_pos_weight to handle severe class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    with mlflow.start_run(run_name="xgboost_fraud_detector"):
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "scale_pos_weight": scale_weight,
            "random_state": 42,
            "n_jobs": -1
        }
        
        mlflow.log_params(params)
        
        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(X_train, y_train)
        
        xgb_preds = xgb_model.predict(X_test)
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        
        xgb_precision = precision_score(y_test, xgb_preds, zero_division=0)
        xgb_recall = recall_score(y_test, xgb_preds, zero_division=0)
        xgb_f1 = f1_score(y_test, xgb_preds, zero_division=0)
        xgb_roc_auc = roc_auc_score(y_test, xgb_probs)
        xgb_pr_auc = average_precision_score(y_test, xgb_probs)
        
        # Log metrics to MLflow
        mlflow.log_metric("precision", xgb_precision)
        mlflow.log_metric("recall", xgb_recall)
        mlflow.log_metric("f1_score", xgb_f1)
        mlflow.log_metric("roc_auc", xgb_roc_auc)
        mlflow.log_metric("pr_auc", xgb_pr_auc)
        
        mlflow.xgboost.log_model(xgb_model, "xgboost_model")
        logging.info(f"XGBoost Metrics -> Precision: {xgb_precision:.4f}, Recall: {xgb_recall:.4f}, ROC-AUC: {xgb_roc_auc:.4f}")

    logging.info("🎉 Model training and MLflow experiment tracking completed successfully!")

if __name__ == "__main__":
    train_models()