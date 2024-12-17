# D100-D400 Project: Used Car Price Prediction

## Overview

This project involves predicting used car prices using advanced data science techniques, including exploratory data analysis (EDA), feature engineering, and machine learning models such as Gradient Boosting and ElasticNet. The repository is designed to be a reusable package with a clear structure and modular components.

---

## Features

- **Exploratory Data Analysis (EDA)**: Tools to explore and clean data.
- **Feature Engineering**: Custom pipelines to preprocess data.
- **Model Training**: Machine learning pipelines for ElasticNet (GLM) and Gradient Boosting (LGBM) regression.
- **Model Evaluation**: Metrics and visualizations for evaluating model performance.
- **Test Suite**: Unit tests for ensuring reliability of custom classes.

---

## Dataset

The dataset for this project is stored in the `data` folder under the file `raw_data.csv`. It contains information about used cars, including:

- **Price**: The price of the car (target variable).
- **Year**: The manufacturing year of the car.
- **Kilometer**: The number of kilometers driven.
- **Fuel Type**: The type of fuel used (e.g., Petrol, Diesel).
- **Transmission**: The transmission type (e.g., Manual, Automatic).
- **Engine**: The engine displacement (e.g., `1198 cc`).
- **Max Power**: The maximum power output of the car (e.g., `87 bhp`).
- **Max Torque**: The maximum torque (e.g., `109 Nm`).
- **Length, Width**: The car's dimensions in millimeters.
- **Fuel Tank Capacity**: The fuel tank's capacity in liters.

The dataset is preprocessed and split during the pipeline stages for model training and evaluation.

---

## Installation

### Using `environment.yml`

1. Ensure Conda is installed. You can download it from [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. Create the environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate d100-d400-project
   ```

4. To update the environment (if the environment.yml changes), use:
   ```bash
   conda env update -f environment.yml --prune
   ```

---

## Repository Structure

- **data/**: Raw and processed data
  - `load_data.py`: Data loading script
  - `raw_data.csv`: Example dataset
- **notebooks/**: Jupyter Notebooks for analysis
  - `1.EDA_Cleaning.ipynb`: EDA and data cleaning steps
  - `2.Load and Split.ipynb`: Data loading and splitting
- **src/**: Source code
  - `__init__.py`: Makes `src` a package
  - `EDA.py`: EDA functions
  - `evaluation.py`: Model evaluation scripts
  - `feature_engineering.py`: Feature engineering pipelines
  - `hyperparameter_tuning.py`: Hyperparameter optimization
  - `model_pipelines.py`: Model training pipelines
- **tests/**: Unit tests for the project
  - `test_feature_engineering.py`: Test cases for feature engineering
- `environment.yml`: Conda environment file
- `README.md`: Project documentation
- `setup.py`: Installable package setup

---

## Running the Project

### 1. Run the Exploratory Data Analysis (EDA)
Execute the EDA notebook to explore and clean your data:
```bash
jupyter notebook notebooks/1.EDA_Cleaning.ipynb
```

### 2. Run Data Cleaning and Split the Data

- After completing the EDA, clean the dataset by handling:
  - Missing values
  - Outliers
  - Irrelevant features

- Then, run the notebook `2.Load and Split.ipynb` to split the data into training and testing sets:
  ```bash
  jupyter notebook notebooks/2.Load and Split.ipynb
  ```
  

### 3. Perform Feature Engineering

Run the feature engineering script in the terminal to preprocess the data:
```bash
python src/feature_engineering.py
```

### 4. Train the Models

Run the `model_pipelines.py` script in the terminal to train the models:
```bash
python src/model_pipelines.py
```

### 5. Tune Hyperparameters

Run the `hyperparameter_tuning.py` script in the terminal to optimize the model performance:
```bash
python src/hyperparameter_tuning.py
```

### 6. Evaluate the Models

Run the `evaluation.py` script in the terminal to assess model performance:
```bash
python src/evaluation.py
```

---

## Dependencies

The dependencies are managed using Conda via the `environment.yml` file. To recreate the environment, follow the steps outlined in the **Installation** section.

---

## Tests

Run the test suite to verify the functionality of the code:
```bash
pytest tests/
```


