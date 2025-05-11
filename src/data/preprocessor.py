import pandas as pd

def analyze_missing_values(df):
    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / len(df)) * 100
    missing_data = pd.DataFrame({
        'Missing Values': missing_counts[missing_counts > 0],
        'Percentage (%)': missing_percentage[missing_counts > 0].round(2)
    })
    
    if missing_data.empty:
        print("❌ No missing values found in the dataset.")
    else:
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


def compute_descriptive_statistics(df):
    return df.describe(include='all')


# Update the function to also print the remaining columns
def drop_irrelevant_columns(df):
    columns_to_drop = ['ID', 'Name', 'Doctor', 'Hospital', 'Room Number','Billing Amount','Discharge Date', 'Date of Admission']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    print("Remaining columns:", df.columns.tolist())
    return df


def handle_missing_values(df):
    missing_summary = df.isnull().sum()
    total_missing = missing_summary.sum()

    if total_missing == 0:
        print("❌ No missing values found.")
    else:
        print("⚠️ Missing values found. Filling them accordingly...")
        for column in df.columns:
            if df[column].isnull().any():
                if df[column].dtype == 'object':
                    # Fill missing categorical values with mode
                    fill_value = df[column].mode()[0]
                    df[column].fillna(fill_value, inplace=True)
                    print(f"Filled missing values in '{column}' with mode: '{fill_value}'")
                else:
                    # Fill missing numerical values with median
                    fill_value = df[column].median()
                    df[column].fillna(fill_value, inplace=True)
                    print(f"Filled missing values in '{column}' with median: {fill_value}")
    return df

