import os
import pandas as pd

# Path to your raw CSV file inside the project
raw_file_path = "ml/data/raw/PS_20174392719_1491204439457_log.csv"

def verify_dataset():
    if not os.path.exists(raw_file_path):
        print(f"❌ Error: Dataset not found at {raw_file_path}. Please check the folder path.")
        return
    
    print("✅ Dataset found! Loading a small sample (50,000 rows) to test...")
    
    # Read just the first 50,000 rows to test quickly without freezing your computer
    df = pd.read_csv(raw_file_path, nrows=50000)
    
    print(f"\n--- Dataset Info ---")
    print(f"Sample Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Fraud cases in sample: {df['isFraud'].sum()} out of {len(df)} rows")
    print("\nData preview loaded successfully!")

if __name__ == "__main__":
    verify_dataset()