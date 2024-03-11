import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def clean_data(
    df, columns=None, handle_missing="mean", convert_dtypes=None, outliers="iqr"
):
    """
    Cleans a pandas DataFrame by handling missing values, converting data types, and handling outliers.

    Args:
        df (pandas.DataFrame): The DataFrame to clean.
        columns (list, optional): List of columns to focus cleaning on. Defaults to None (all columns).
        handle_missing (str, optional): Method to handle missing values (e.g., "mean", "median", "drop"). Defaults to "mean".
        convert_dtypes (dict, optional): Dictionary mapping columns to desired data types. Defaults to None.
        outliers (str, optional): Method to handle outliers ("iqr" for Interquartile Range, custom function possible). Defaults to "iqr".

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    if columns is None:
        columns = df.columns
    for col in columns:
        if handle_missing == "mean":
            df[col].fillna(df[col].mean(), inplace=True)  # Replace with mean
        elif handle_missing == "median":
            df[col].fillna(df[col].median(), inplace=True)  # Replace with median
        elif handle_missing == "drop":
            df.dropna(
                subset=[col], inplace=True
            )  # Drop rows with missing values in this column
        # Add more options for handling missing values (e.g., custom function)
        if convert_dtypes:
            df[col] = df[col].astype(convert_dtypes[col])  # Convert data type
        if outliers == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (1.5 * IQR)
            upper_bound = Q3 + (1.5 * IQR)
            df.drop(
                df[col < lower_bound].index, inplace=True
            )  # Remove outliers below IQR threshold
            df.drop(
                df[col > upper_bound].index, inplace=True
            )  # Remove outliers above IQR threshold
        # Add option for custom outlier handling function

    return df
