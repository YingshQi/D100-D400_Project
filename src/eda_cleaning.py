import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#EDA 
def describe_data(df):
    """
    Prints basic information and statistics of the DataFrame.
    """
    print("Dataset Information:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe(include="all"))


def plot_numeric_distribution(df, column):
    """
    Plots the distribution of a numerical column.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name of the numerical variable to plot.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


def plot_categorical_distribution(df, column):
    """
    Plots the distribution of a categorical column.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name of the categorical variable to plot.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(y=df[column])
    plt.title(f'Distribution of {column}')
    plt.ylabel(column)
    plt.xlabel('Count')
    plt.show()


def visualize_missing_values(df):
    """
    Visualizes missing values in the DataFrame.
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title('Missing Values Heatmap')
    plt.show()


def summarize_missing_values(df):
    """
    Prints the count of missing values for each column.
    """
    print("Missing Values:")
    print(df.isnull().sum())


def plot_boxplot(df, column):
    """
    Plots a boxplot for detecting outliers in a numerical column.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name of the numerical variable to plot.
    """
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)
    plt.show()


def plot_correlation_heatmap(df):
    """
    Plots a heatmap of the correlation matrix for numeric columns.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
    """
    numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title('Correlation Heatmap')
    plt.show()


def plot_feature_relationship(df, x_column, y_column):
    """
    Plots a scatterplot for the relationship between two features.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_column (str): Column name for the x-axis.
        y_column (str): Column name for the y-axis.
    """
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[x_column], y=df[y_column])
    plt.title(f'{x_column} vs. {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

#data cleaning
"""
def handle_missing_values(df):
    
    Handles missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame with missing values.
    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    
    if 'Engine' in df.columns:
        df['Engine'] = df['Engine'].str.replace('cc', '').astype(float)
        df['Engine'].fillna(df['Engine'].median(), inplace=True)
    
    for column in ['Length', 'Width', 'Height']:
        if column in df.columns:
            df[column].fillna(df[column].median(), inplace=True)
    
    # Drop rows with missing 'Price' (target variable)
    df.dropna(subset=['Price'], inplace=True)
    
    return df
"""
def handle_missing_values(df):
    """
    Handles missing values in the dataset.
    Args:
        df (pd.DataFrame): Input DataFrame with missing values.
    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    # Impute numeric columns with their median
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numeric_columns:
        df[column].fillna(df[column].median(), inplace=True)

    # Impute categorical columns with their mode (most frequent value)
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column].fillna(df[column].mode()[0], inplace=True)

    return df

def convert_and_clean_numeric_columns(df):
    """
    Converts and cleans numeric columns.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    # Remove non-numeric characters from the 'Engine' column and convert to float
    if 'Engine' in df.columns:
        df['Engine'] = df['Engine'].str.replace('cc', '', regex=False).astype(float)

    return df


def add_car_age(df):
    """
    Adds a new feature for car age based on the 'Year' column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with 'Car Age' column added.
    """
    if 'Year' in df.columns:
        df['Car Age'] = 2024 - df['Year']
        df.drop(columns=['Year'], inplace=True)  # Drop 'Year' after creating 'Car Age'
    return df



def remove_outliers(df, column):
    """
    Removes outliers in the specified column using the IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name for which outliers are to be removed.
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]


def encode_categorical_variables(df):
    """
    Encodes categorical variables using one-hot encoding.

    Args:
        df (pd.DataFrame): Input DataFrame with categorical variables.

    Returns:
        pd.DataFrame: DataFrame with categorical variables encoded.
    """
    categorical_columns = ['Name', 'Fuel Type', 'Transmission', 'Seller Type', 'Owner', 'Drivetrain', 'Location', 'Color']
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)





def normalize_numerical_features(df, columns):
    """
    Normalizes numerical features to have values between 0 and 1.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of columns to normalize.
    Returns:
        pd.DataFrame: DataFrame with normalized numerical features.
    """
    for column in columns:
        if column in df.columns:
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

def impute_categorical_values(df):
    """
    Imputes missing categorical values with the most frequent value.
    
    Args:
        df (pd.DataFrame): Input DataFrame with missing categorical values.
    Returns:
        pd.DataFrame: DataFrame with imputed categorical values.
    """
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
    return df

def remove_duplicates(df):
    """
    Removes duplicate rows from the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates()
