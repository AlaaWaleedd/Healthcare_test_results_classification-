import pandas as pd

def analyze_missing_values(df):
    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / len(df)) * 100
    missing_data = pd.DataFrame({
        'Missing Values': missing_counts[missing_counts > 0],
        'Percentage (%)': missing_percentage[missing_counts > 0].round(2)
    })
    missing_data = missing_data.sort_values('Percentage (%)', ascending=False)
    print("Missing Values Analysis:")
    print(missing_data)
    return missing_data

def get_numerical_features(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['ID']

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    filtered = [col for col in num_cols if col not in exclude_cols]
    return filtered

def get_plotable_categorical_features(df, target_col=None, exclude_cols=None, max_unique=20):
    if exclude_cols is None:
        exclude_cols = ['ID', 'Name', 'Doctor', 'Hospital', 'Medication', 'Date of Admission', 'Discharge Date']

    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    plotable = [
        col for col in categorical_cols
        if col not in exclude_cols
        and (target_col is None or col != target_col)
        and df[col].nunique() <= max_unique
    ]

    return plotable


def get_monthly_test_result_counts(df, date_col='Date of Admission', target_col='Test Results'):
    """
    Groups test results by year-month from the date column and returns counts.

    Args:
        df (pd.DataFrame): Input DataFrame.
        date_col (str): Column containing admission dates.
        target_col (str): Target classification column (e.g., 'Test Results').

    Returns:
        pd.DataFrame: Grouped counts by month and target class.
    """
    # Ensure date column is in datetime format
    df['Month'] = pd.to_datetime(df[date_col], errors='coerce').dt.strftime('%Y-%m')

    # Group by month and target column
    grouped_data = (
        df.groupby(['Month', target_col])
          .size()
          .reset_index(name='Count')
          .sort_values(by='Month')
    )

    return grouped_data
