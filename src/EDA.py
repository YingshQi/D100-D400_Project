import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def describe_data(df):
    """
    Describes the dataset with basic information and statistics.
    """
    print("Dataset Info:")
    print(df.info())
    print("\nDescriptive Statistics:")
    return df.describe(include="all")

def check_missing_values(df):
    """
    Checks and visualizes missing values in the dataset.
    """
    missing_values = df.isnull().sum()
    print("\nMissing Values Per Column:")
    print(missing_values)
    
    # Visualize missing values
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.show()
    return missing_values

def plot_target_distribution(df, target_col):
    """
    Plots the distribution of the target variable.
    """
    if target_col in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[target_col], kde=True, bins=30)
        plt.title(f"Distribution of {target_col}")
        plt.xlabel(target_col)
        plt.ylabel("Frequency")
        plt.show()
    else:
        print(f"The column '{target_col}' does not exist in the dataset.")

def detect_outliers(df, col):
    """
    Detects outliers in a specific column using boxplot.
    """
    if col in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[col])
        plt.title(f"Outliers in {col}")
        plt.xlabel(col)
        plt.show()
    else:
        print(f"The column '{col}' does not exist in the dataset.")

def correlation_analysis(df, target_col):
    """
    Analyzes and visualizes correlations with the target variable.
    """
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if target_col in numeric_cols:
        correlations = df[numeric_cols].corr()[target_col].sort_values(ascending=False)
        print(f"\nCorrelations with {target_col}:")
        print(correlations)
        
        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()
        return correlations
    else:
        print(f"The target column '{target_col}' is not numeric or does not exist.")
        return None
