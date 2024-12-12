# Import necessary libraries
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from pathlib import Path


# Dynamically locate the project root based on this file's location
project_root = Path().resolve().parent
sys.path.append(str(project_root))

data_path = project_root / "data" / "processed_data.parquet"

# Step 1: Load cleaned data
print("Loading cleaned data...")
df = pd.read_parquet(data_path)
print(f"Cleaned data loaded successfully with shape: {df.shape}")

# Step 2: Define features (X) and target (y)
print("Splitting data into features (X) and target (y)...")
target_column = "Price"  # Replace with the correct target column if different
X = df.drop(columns=[target_column])  # Features
y = df[target_column]  # Target variable
print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# Step 3: Split the dataset into training and testing sets
print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 20% for testing
)
print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

# Save the split datasets (optional, for reproducibility)
output_dir = project_root / "data"
print(f"Saving split datasets to {output_dir}...")
X_train.to_parquet(output_dir / "X_train.parquet")
X_test.to_parquet(output_dir / "X_test.parquet")
y_train.to_frame(name=target_column).to_parquet(output_dir / "y_train.parquet")
y_test.to_frame(name=target_column).to_parquet(output_dir / "y_test.parquet")
print("Datasets saved successfully.")

# Optional: Inspect first few rows of the training set
print("First few rows of training features:")
print(X_train.head())
