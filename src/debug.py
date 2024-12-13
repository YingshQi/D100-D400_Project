import os
from pathlib import Path
import pandas as pd

print("DEBUGGING SCRIPT STARTED")

# Check the current working directory
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

# Check paths
project_root = Path().resolve().parent
print(f"Project root: {project_root}")

data_path = project_root / "data"
print(f"Data path: {data_path}")

# Check if data files exist
processed_file = data_path / "processed_data.parquet"
print(f"Checking if processed_data.parquet exists: {processed_file.exists()}")

if processed_file.exists():
    print("File exists. Loading...")
    try:
        df = pd.read_parquet(processed_file)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading parquet file: {e}")
else:
    print("Processed data file not found!")

print("DEBUGGING SCRIPT COMPLETED")
