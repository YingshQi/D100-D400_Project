# Import necessary libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Import pipelines if defined in another file
from model_pipelines import glm_pipeline, lgbm_pipeline  # Replace 'model_pipelines' with the correct file name if needed

# Define paths
project_root = Path().resolve().parent  # Adjust as needed
data_path = project_root / "data"

# Load the validation set
X_test = pd.read_parquet(data_path / "X_test_transformed.parquet")
y_test = pd.read_parquet(data_path / "y_test.parquet")

# Flatten y_test
y_test = y_test.values.ravel()

# Make predictions on the validation set
y_pred_glm_log = glm_pipeline.predict(X_test)
y_pred_lgbm_log = lgbm_pipeline.predict(X_test)

# Reverse log transformation with clipping
y_test_clipped = np.clip(y_test, None, 20)  # Upper limit to prevent overflow
y_test_original = np.expm1(y_test_clipped)

y_pred_glm_clipped = np.clip(y_pred_glm_log, None, 20)
y_pred_glm = np.expm1(y_pred_glm_clipped)

y_pred_lgbm_clipped = np.clip(y_pred_lgbm_log, None, 20)
y_pred_lgbm = np.expm1(y_pred_lgbm_clipped)

# Evaluate GLM
glm_mse = mean_squared_error(y_test_original, y_pred_glm)
glm_mae = mean_absolute_error(y_test_original, y_pred_glm)
glm_r2 = r2_score(y_test_original, y_pred_glm)
print(f"GLM Evaluation:")
print(f"- Mean Squared Error (MSE): {glm_mse}")
print(f"- Mean Absolute Error (MAE): {glm_mae}")
print(f"- R-squared (R2): {glm_r2}")

# Evaluate LGBM
lgbm_mse = mean_squared_error(y_test_original, y_pred_lgbm)
lgbm_mae = mean_absolute_error(y_test_original, y_pred_lgbm)
lgbm_r2 = r2_score(y_test_original, y_pred_lgbm)
print(f"\nLGBM Evaluation:")
print(f"- Mean Squared Error (MSE): {lgbm_mse}")
print(f"- Mean Absolute Error (MAE): {lgbm_mae}")
print(f"- R-squared (R2): {lgbm_r2}")


# Create predicted vs. actual plot for GLM
plt.figure(figsize=(8, 6))
plt.scatter(y_test_original, y_pred_glm, alpha=0.6, label='GLM Predictions', color='blue')
plt.plot([min(y_test_original), max(y_test_original)], [min(y_test_original), max(y_test_original)], color='red', linestyle='--', label='Perfect Fit')
plt.title("Predicted vs Actual: GLM")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()

# Create predicted vs. actual plot for LGBM
plt.figure(figsize=(8, 6))
plt.scatter(y_test_original, y_pred_lgbm, alpha=0.6, label='LGBM Predictions', color='green')
plt.plot([min(y_test_original), max(y_test_original)], [min(y_test_original), max(y_test_original)], color='red', linestyle='--', label='Perfect Fit')
plt.title("Predicted vs Actual: LGBM")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()
