import pandas as pd
from pathlib import Path

def load_data(file_name: str = "raw_data.csv") -> pd.DataFrame:
    """
    Loads the raw dataset from the data/raw/ directory.

    Parameters:
    ----------
    file_name : str
        Name of the dataset file (default: "raw_data.csv").

    Returns:
    -------
    pd.DataFrame
        The loaded dataset.
    """
    # Resolve the path to the raw data folder
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / file_name

    if not data_path.exists():
        raise FileNotFoundError(f"The file {data_path} does not exist.")
    
    return pd.read_csv(data_path)