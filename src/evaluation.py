
# Import necessary libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from model_pipelines import glm_pipeline, lgbm_pipeline

# Define paths
project_root = Path().resolve().parent
data_path = project_root / "data"

# Load the validation set
X_test = pd.read_parquet(data_path / "X_test_transformed.parquet")
y_test = pd.read_parquet(data_path / "y_test.parquet")

# Flatten y_test
y_test = y_test.values.ravel()

# Reverse log transformation if necessary
log_transformed = True
if log_transformed:
    y_test_original = np.expm1(y_test)
else:
    y_test_original = y_test

# Make predictions
y_pred_glm = glm_pipeline.predict(X_test)
y_pred_lgbm = lgbm_pipeline.predict(X_test)

if log_transformed:
    y_pred_glm = np.expm1(y_pred_glm)
    y_pred_lgbm = np.expm1(y_pred_lgbm)

# Evaluate GLM
glm_mse = mean_squared_error(y_test_original, y_pred_glm)
glm_mae = mean_absolute_error(y_test_original, y_pred_glm)
glm_mape = mean_absolute_percentage_error(y_test_original, y_pred_glm)
glm_r2 = r2_score(y_test_original, y_pred_glm)

# Evaluate LGBM
lgbm_mse = mean_squared_error(y_test_original, y_pred_lgbm)
lgbm_mae = mean_absolute_error(y_test_original, y_pred_lgbm)
lgbm_mape = mean_absolute_percentage_error(y_test_original, y_pred_lgbm)
lgbm_r2 = r2_score(y_test_original, y_pred_lgbm)

# Print evaluation metrics
print("GLM Evaluation:")
print(f"Mean Squared Error (MSE): {glm_mse}")
print(f"Mean Absolute Error (MAE): {glm_mae}")
print(f"Mean Absolute Percentage Error (MAPE): {glm_mape:.2%}")
print(f"R-squared (R2): {glm_r2}")

print("\nLGBM Evaluation:")
print(f"Mean Squared Error (MSE): {lgbm_mse}")
print(f"Mean Absolute Error (MAE): {lgbm_mae}")
print(f"Mean Absolute Percentage Error (MAPE): {lgbm_mape:.2%}")
print(f"R-squared (R2): {lgbm_r2}")

# Create "predicted vs. actual" plots
plt.figure(figsize=(8, 6))
plt.scatter(y_test_original, y_pred_glm, alpha=0.6, label="GLM Predictions", color="blue")
plt.plot([min(y_test_original), max(y_test_original)], [min(y_test_original), max(y_test_original)], color="red", linestyle="--", label="Perfect Fit")
plt.title("Predicted vs Actual: GLM")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.savefig("glm_predicted_vs_actual.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(y_test_original, y_pred_lgbm, alpha=0.6, label="LGBM Predictions", color="green")
plt.plot([min(y_test_original), max(y_test_original)], [min(y_test_original), max(y_test_original)], color="red", linestyle="--", label="Perfect Fit")
plt.title("Predicted vs Actual: LGBM")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.savefig("lgbm_predicted_vs_actual.png")
plt.close()

# Identify top 5 most important features
feature_importances = lgbm_pipeline.named_steps['regressor'].feature_importances_
top_features_idx = np.argsort(feature_importances)[-5:][::-1]
top_features = X_test.columns[top_features_idx]

print("\nTop 5 Most Relevant Features (LGBM):")
for idx, feature in enumerate(top_features, start=1):
    print(f"{idx}. {feature} (Importance: {feature_importances[top_features_idx[idx-1]]})")

# Partial Dependence Plots for LGBM
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(top_features):
    PartialDependenceDisplay.from_estimator(
        lgbm_pipeline,
        X_test,
        features=[feature],
        grid_resolution=20,
        ax=axes[i]
    )
plt.tight_layout()
plt.savefig("partial_dependence_lgbm.png")
plt.close()


"""
# Import necessary libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from model_pipelines import glm_pipeline, lgbm_pipeline

# Define paths
project_root = Path().resolve().parent
data_path = project_root / "data"

# Load the validation set
X_test = pd.read_parquet(data_path / "X_test_transformed.parquet")
y_test = pd.read_parquet(data_path / "y_test.parquet")

# Flatten y_test
y_test = y_test.values.ravel()

# Reverse log transformation if necessary
log_transformed = True
if log_transformed:
    y_test_original = np.expm1(y_test)
else:
    y_test_original = y_test

# Make predictions
y_pred_glm = glm_pipeline.predict(X_test)
y_pred_lgbm = lgbm_pipeline.predict(X_test)

if log_transformed:
    y_pred_glm = np.expm1(y_pred_glm)
    y_pred_lgbm = np.expm1(y_pred_lgbm)

# Evaluate GLM
glm_mse = mean_squared_error(y_test_original, y_pred_glm)
glm_mae = mean_absolute_error(y_test_original, y_pred_glm)
glm_r2 = r2_score(y_test_original, y_pred_glm)

# Evaluate LGBM
lgbm_mse = mean_squared_error(y_test_original, y_pred_lgbm)
lgbm_mae = mean_absolute_error(y_test_original, y_pred_lgbm)
lgbm_r2 = r2_score(y_test_original, y_pred_lgbm)

# Print evaluation metrics
print("GLM Evaluation:")
print(f"Mean Squared Error (MSE): {glm_mse}")
print(f"Mean Absolute Error (MAE): {glm_mae}")
print(f"R-squared (R2): {glm_r2}")

print("\nLGBM Evaluation:")
print(f"Mean Squared Error (MSE): {lgbm_mse}")
print(f"Mean Absolute Error (MAE): {lgbm_mae}")
print(f"R-squared (R2): {lgbm_r2}")

# Create "predicted vs. actual" plots
plt.figure(figsize=(8, 6))
plt.scatter(y_test_original, y_pred_glm, alpha=0.6, label="GLM Predictions", color="blue")
plt.plot([min(y_test_original), max(y_test_original)], [min(y_test_original), max(y_test_original)], color="red", linestyle="--", label="Perfect Fit")
plt.title("Predicted vs Actual: GLM")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.savefig("glm_predicted_vs_actual.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(y_test_original, y_pred_lgbm, alpha=0.6, label="LGBM Predictions", color="green")
plt.plot([min(y_test_original), max(y_test_original)], [min(y_test_original), max(y_test_original)], color="red", linestyle="--", label="Perfect Fit")
plt.title("Predicted vs Actual: LGBM")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.savefig("lgbm_predicted_vs_actual.png")
plt.close()

# Identify top 5 most important features
feature_importances = lgbm_pipeline.named_steps['regressor'].feature_importances_
top_features_idx = np.argsort(feature_importances)[-5:][::-1]
top_features = X_test.columns[top_features_idx]

print("\nTop 5 Most Relevant Features (LGBM):")
for idx, feature in enumerate(top_features, start=1):
    print(f"{idx}. {feature} (Importance: {feature_importances[top_features_idx[idx-1]]})")

# Partial Dependence Plots for LGBM
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(top_features):
    PartialDependenceDisplay.from_estimator(
        lgbm_pipeline,
        X_test,
        features=[feature],
        grid_resolution=20,
        ax=axes[i]
    )
plt.tight_layout()
plt.savefig("partial_dependence_lgbm.png")
plt.close()
"""