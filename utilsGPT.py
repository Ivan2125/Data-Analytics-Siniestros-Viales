# utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def load_data(filepath, data_type="csv", sep=",", encoding="utf-8", low_memory=False):
    """
    Loads data from a file into a pandas DataFrame. Supports various data types and options.

    Args:
        filepath (str): Path to the data file.
        data_type (str, optional): Type of data file. Defaults to "csv".
        sep (str, optional): Delimiter for separated data (e.g., comma for CSV). Defaults to ",".
        encoding (str, optional): Encoding of the data file. Defaults to "utf-8".
        low_memory (bool, optional): Read data in chunks to handle large files. Defaults to False.

    Returns:
        pandas.DataFrame: The loaded data as a DataFrame.
    """
    if data_type == "csv":
        return pd.read_csv(filepath, sep=sep, encoding=encoding, low_memory=low_memory)
    # Add support for other data types (e.g., excel, json) as needed
    else:
        raise NotImplementedError(f"Data type '{data_type}' not supported yet.")


def perform_eda(df, target_col=None):
    """
    Perform comprehensive Exploratory Data Analysis (EDA) on the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame for EDA.
    - target_col (str, optional): Target column for analysis (for classification problems).
    """
    # Display basic information about the DataFrame
    print("Data Overview:")
    print(df.info())

    # Display summary statistics
    print("\nSummary Statistics:")
    print(df.describe())

    # Display missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Display correlation matrix and heatmap
    print("\nCorrelation Matrix:")
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title("Correlation Matrix")
    plt.show()

    # Pairplot for numeric columns
    print("\nPairplot for Numeric Columns:")
    sns.pairplot(df, hue=target_col, diag_kind="kde")
    plt.show()

    # Boxplot for each numeric column
    print("\nBoxplot for Numeric Columns:")
    numeric_cols = df.select_dtypes(include=np.number).columns
    plt.figure(figsize=(15, 8))
    for i, col in enumerate(numeric_cols, start=1):
        plt.subplot(2, len(numeric_cols) // 2, i)
        sns.boxplot(x=target_col, y=col, data=df)
        plt.title(f"Boxplot for {col}")
    plt.tight_layout()
    plt.show()

    # Countplot for categorical columns
    print("\nCountplot for Categorical Columns:")
    cat_cols = df.select_dtypes(include="object").columns
    plt.figure(figsize=(15, 8))
    for i, col in enumerate(cat_cols, start=1):
        plt.subplot(2, len(cat_cols) // 2, i)
        sns.countplot(x=col, data=df, hue=target_col)
        plt.title(f"Countplot for {col}")
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def clean_data(df, numeric_cols=None, categorical_cols=None, target_col=None):
    """
    Clean the DataFrame by handling missing values, outliers, etc.

    Parameters:
    - df (pd.DataFrame): DataFrame to be cleaned.
    - numeric_cols (list, optional): List of numeric columns for scaling.
    - categorical_cols (list, optional): List of categorical columns for encoding.
    - target_col (str, optional): Target column for analysis (for classification problems).

    Returns:
    - cleaned_df (pd.DataFrame): Cleaned DataFrame.
    """
    # Handle missing values (you can customize this based on your data)
    cleaned_df = df.dropna()

    # Handle outliers (you can customize this based on your data)
    # Example: Remove rows where a specific column has values outside a certain range
    cleaned_df = cleaned_df[
        (cleaned_df["numeric_column"] >= min_value)
        & (cleaned_df["numeric_column"] <= max_value)
    ]

    # Standardize numeric columns
    if numeric_cols:
        scaler = StandardScaler()
        cleaned_df[numeric_cols] = scaler.fit_transform(cleaned_df[numeric_cols])

    # One-hot encode categorical columns
    if categorical_cols:
        cleaned_df = pd.get_dummies(
            cleaned_df, columns=categorical_cols, drop_first=True
        )

    return cleaned_df


def save_data(df, output_path):
    """
    Save the DataFrame to a CSV file.

    Parameters:
    - df (pd.DataFrame): DataFrame to be saved.
    - output_path (str): Path to save the CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")


def explore_data(df, categorical_plots=True, numerical_plots=True):
    """
    Performs exploratory data analysis on a pandas DataFrame with visualizations.

    Args:
        df (pandas.DataFrame): The DataFrame to explore.
        categorical_plots (bool, optional): Whether to generate plots for categorical columns. Defaults to True.
        numerical_plots (bool, optional): Whether to generate plots for numerical columns. Defaults to True.
    """
    print(df.describe(include="all"))
    for col in df.select_dtypes(include=[object]):  # Explore categorical columns
        if categorical_plots:
            sns.countplot(x=col, data=df)
            plt.show()  # Explicitly show plots
    for col in df.select_dtypes(include=[int64, float64]):  # Explore numerical columns
        if numerical_plots:
            sns
