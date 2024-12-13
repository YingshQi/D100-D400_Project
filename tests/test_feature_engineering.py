import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Locate project root dynamically
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]  # Go two levels up
src_path = project_root / "src"

# Add src folder to sys.path
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"'src' directory not found at: {src_path}")

# Import the classes to be tested
try:
    from feature_engineering import CustomStandardScaler, CustomOneHotEncoder
except ImportError as e:
    raise ImportError(f"Failed to import classes from feature_engineering.py: {e}")


# Test CustomStandardScaler
@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        # Updated expected outputs with precise values
        (pd.DataFrame({"col1": [1, 2, 3]}), pd.DataFrame({"col1": [-1.224744871391589, 0.0, 1.224744871391589]})),
        (pd.DataFrame({"col1": [1, 2, 3], "col2": [10, 20, 30]}),
         pd.DataFrame({"col1": [-1.224744871391589, 0.0, 1.224744871391589],
                       "col2": [-1.224744871391589, 0.0, 1.224744871391589]})),
        (pd.DataFrame({"col1": [5, 5, 5]}), pd.DataFrame({"col1": [0.0, 0.0, 0.0]})),  # Handle zero std
    ],
)
def test_custom_standard_scaler(input_data, expected_output):
    scaler = CustomStandardScaler()
    transformed = scaler.fit_transform(input_data)
    pd.testing.assert_frame_equal(
        pd.DataFrame(transformed, columns=input_data.columns),
        expected_output,
        atol=1e-6,  # Relaxed tolerance for floating-point comparisons
    )


# Test CustomOneHotEncoder
@pytest.mark.parametrize(
    "input_data, fit_data, expected_columns, expected_output",
    [
        (pd.DataFrame({"col1": ["A", "B", "A"]}),
         pd.DataFrame({"col1": ["A", "B"]}),
         ["col1_A", "col1_B"],
         [[1, 0], [0, 1], [1, 0]]),
        (pd.DataFrame({"col1": ["A", "B", "A"], "col2": ["X", "Y", "X"]}),
         pd.DataFrame({"col1": ["A", "B"], "col2": ["X", "Y"]}),
         ["col1_A", "col1_B", "col2_X", "col2_Y"],
         [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]),
        (pd.DataFrame({"col1": ["A", "C"]}),
         pd.DataFrame({"col1": ["A", "B"]}),
         ["col1_A", "col1_B"],
         [[1, 0], [0, 0]]),  # "C" is ignored due to handle_unknown='ignore'
    ],
)
def test_custom_one_hot_encoder(input_data, fit_data, expected_columns, expected_output):
    encoder = CustomOneHotEncoder()
    encoder.fit(fit_data)  # Fit on specified training data
    transformed = encoder.transform(input_data)
    # Ensure the transformed columns match expected columns
    assert list(transformed.columns) == expected_columns
    # Ensure the transformed data matches expected output
    np.testing.assert_array_equal(transformed.values, np.array(expected_output))
