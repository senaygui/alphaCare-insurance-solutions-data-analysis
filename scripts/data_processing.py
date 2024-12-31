import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def check_missing_values(df):
    """
    Check for missing values in the dataset and return a summary.

    Parameters:
        df (DataFrame): The dataset to check.

    Returns:
        DataFrame: A summary table with missing values, percentages, and column data types.
    """
    # Calculate missing values and percentages
    missing_values = df.isnull().sum()
    missing_percentages = 100 * missing_values / len(df)

    # Filter to only columns with missing values
    missing_values = missing_values[missing_values > 0]
    missing_percentages = missing_percentages[missing_percentages > 0]

    # Ensure the index aligns before concatenation
    column_data_types = df.dtypes[missing_values.index]

    # Create the summary DataFrame
    summary = pd.concat([missing_values, missing_percentages, column_data_types], axis=1)
    summary.columns = ['Missing Values', 'Percentage (%)', 'Data Type']

    return summary

def drop_high_missing_columns(df, threshold=50):
    """
    Drop columns with missing values above the specified threshold.
    
    :param df: pandas DataFrame
    :param threshold: percentage threshold for dropping columns (default 50%)
    :return: DataFrame with high-missing columns dropped
    """
    missing_series = df.isnull().sum() / len(df) * 100
    columns_to_drop = missing_series[missing_series > threshold].index
    df_cleaned = df.drop(columns=columns_to_drop)
    print(f"Dropped columns: {list(columns_to_drop)}")
    return df_cleaned

def handle_missing_values(df, method='mean'):
    """
    Handle missing values in the dataset using the specified method.

    Parameters:
        df (DataFrame): The dataset to handle missing values.
        method (str): The method to use for filling missing values.
            Options: 'mean', 'median', 'mode', or a specific value.

    Returns:
        DataFrame: The dataset with missing values filled.
    """
    for column in df.columns:
        if df[column].dtype == 'object':
            # Categorical column: impute with mode
            mode_value = df[column].mode()[0]
            df[column] = df[column].fillna(mode_value)
        else:
            # Numerical column: handle based on method
            if method == 'mean':
                df[column] = df[column].fillna(df[column].mean())
            elif method == 'median':
                df[column] = df[column].fillna(df[column].median())
            elif method == 'mode':
                df[column] = df[column].fillna(df[column].mode()[0])
            else:
                # Use the specified value
                df[column] = df[column].fillna(method)

    return df

# Function to plot histogram for numerical columns
def plot_histogram(df, column, color=None):
    """
    Plot a histogram for the specified numerical column.
    Parameters:
        df (DataFrame): The dataset.
        column (str): The column to plot.
        color (str): The color for the histogram bars. Default is None.
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, color=color)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Function to plot bar chart for categorical columns with a color parameter
def plot_bar_chart(df, column, color=None):
    """
    Plot a bar chart for the specified categorical column.
    Parameters:
        df (DataFrame): The dataset.
        column (str): The column to plot.
        color (str): The color for the bars. Default is None.
    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    df[column].value_counts().plot(kind='bar', color=color)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

    # function to perform correlation heatmap for key numerical columns
def plot_correlation_heatmap(df, columns):
    corr = df[columns].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='viridis', fmt='.2f', square=True)
    plt.title('Correlation Heatmap')
    plt.show()

def plot_scatter(df, x, y, hue=None):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x=x, y=y, hue=hue)
    plt.title(f'{y} vs {x}')
    plt.show()

def plot_boxplot(df, x, y):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=x, y=y)
    plt.title(f'Boxplot of {y} by {x}')
    plt.xticks(rotation=45)
    plt.show()

def plot_countplot(df, x, hue=None):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x=x, hue=hue)
    plt.title(f'Countplot of {x} by {hue}')
    plt.xticks(rotation=45)
    plt.show()

def plot_pairplot(df, columns, hue=None):
    plt.figure(figsize=(12, 10))
    sns.pairplot(df[columns], hue=hue, palette='viridis')
    plt.show()

def handle_outliers(df):
    """
    Handle outliers in the dataset using the IQR method.

    Parameters:
        df (DataFrame): The dataset to handle outliers.
        columns (list or None): List of columns to check for outliers. 
                                If None, numerical columns are selected automatically.

    Returns:
        DataFrame: The dataset with outliers handled.
    """
    import numpy as np  # Ensure numpy is imported

    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df_cleaned = df.copy()

    # Automatically detect numeric columns if no columns are provided
    columns = df_cleaned.select_dtypes(include=[np.number]).columns

    # Iterate over each specified column and handle outliers
    for column in columns:
        if column in df_cleaned.columns:  # Ensure the column exists in the DataFrame
            Q1 = df_cleaned[column].quantile(0.25)  # First quartile
            Q3 = df_cleaned[column].quantile(0.75)  # Third quartile
            IQR = Q3 - Q1  # Interquartile range
            lower_bound = Q1 - 1.5 * IQR  # Lower bound
            upper_bound = Q3 + 1.5 * IQR  # Upper bound

            # Cap values outside the bounds
            df_cleaned[column] = np.where(
                df_cleaned[column] < lower_bound, lower_bound, df_cleaned[column]
            )
            df_cleaned[column] = np.where(
                df_cleaned[column] > upper_bound, upper_bound, df_cleaned[column]
            )

    return df_cleaned


# outlier plot for each individual numeric column
def outlier_box_plots(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[column])
        plt.title(f'Box plot of {column}')
        plt.show()

def trend_over_geography(df, geography_column, value_column):
    grouped = df.groupby(geography_column)[value_column].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    grouped.plot(kind='bar')
    plt.title(f'Average {value_column} by {geography_column}')
    plt.xticks(rotation=45)
    plt.show()