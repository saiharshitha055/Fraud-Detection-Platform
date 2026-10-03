import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_raw_data(filepath: str) -> pd.DataFrame:
    logging.info(f"Loading raw dataset from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset not found at {filepath}")
    return pd.read_csv(filepath)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting feature engineering pipeline...")
    
    # 1. Balance discrepancy features (Business logic: detects forced balance drainage)
    df['errorBalanceOrg'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    # 2. Transaction type encoding (One-hot encoding for categorical variables)
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    
    # 3. Interaction / Risk ratios
    df['amount_to_org_balance'] = df['amount'] / (df['oldbalanceOrg'] + 1.0) # Avoid division by zero
    
    # 4. Drop columns that cause data leakage or are high-cardinality strings
    drop_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    logging.info("Feature engineering completed successfully.")
    return df

def time_based_train_test_split(df: pd.DataFrame, test_size_steps: int = 168):
    """
    Splits data chronologically based on the 'step' column (1 step = 1 hour).
    Default test_size_steps = 168 hours (last 7 days of the simulation).
    """
    logging.info("Performing chronological time-based train-test split...")
    max_step = df['step'].max()
    split_step = max_step - test_size_steps
    
    train_df = df[df['step'] <= split_step]
    test_df = df[df['step'] > split_step]
    
    X_train = train_df.drop(columns=['isFraud'])
    y_train = train_df['isFraud']
    
    X_test = test_df.drop(columns=['isFraud'])
    y_test = test_df['isFraud']
    
    logging.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Test the pipeline on a sample to verify it works seamlessly
    path = "ml/data/raw/PS_20174392719_1491204439457_log.csv"
    raw_df = load_raw_data(path)
    processed_df = engineer_features(raw_df)
    X_train, X_test, y_train, y_test = time_based_train_test_split(processed_df)
    print("✅ Feature engineering script executed successfully!")
    print(f"Features generated: {list(X_train.columns)}")