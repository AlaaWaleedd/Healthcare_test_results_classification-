import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#===========Exploration Functions=================

def analyze_missing_values(df):
    # Work only on the original columns to avoid derived columns like 'month'
    original_columns = df.columns.tolist()
    
    # Calculate missing values and percentages
    missing_data = df[original_columns].isnull().sum()
    total_rows = df.shape[0]
    missing_summary = pd.DataFrame({
        'Missing Values': missing_data,
        'Missing Percentage (%)': (missing_data / total_rows) * 100
    })
    
    # Filter only columns with missing values
    missing_summary = missing_summary[missing_summary['Missing Values'] > 0]
    
    # Sort by highest percentage of missing values
    return missing_summary.sort_values(by='Missing Percentage (%)', ascending=False)




def detect_outliers(df, columns=None, method='iqr', threshold=1.5, z_thresh=3.0, return_summary=False):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_df = pd.DataFrame(False, index=df.index, columns=columns)
    summary = {}

    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df[col] < lower_bound) | (df[col] > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            mask = pd.Series(z_scores > z_thresh, index=df[col].dropna().index)
            mask = mask.reindex(df.index, fill_value=False)

        else:
            raise ValueError("Method must be either 'iqr' or 'zscore'.")

        outlier_df[col] = mask
        summary[col] = mask.sum()

    return (outlier_df, summary) if return_summary else outlier_df


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

def encode_students_dataset(df):
    # Encode categorical columns using one-hot encoding
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def encoding_features(df):
    # You can customize this if needed
    return encode_students_dataset(df)

def save_processed_df(df, filename, output_dir="data/processed"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, filename), index=False)







#===========Preprocessing Functions====================

def handle_missing_values(df):
  
    filled_info = []

    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                filled_info.append(f"Filled numerical column '{col}' with median: {median_val}")
            else:
                mode_val = df[col].mode().dropna()
                if not mode_val.empty:
                    mode_val = mode_val[0]
                    df[col] = df[col].fillna(mode_val)

                    filled_info.append(f"Filled categorical column '{col}' with mode: {mode_val}")
                else:
                    filled_info.append(f"Could not fill column '{col}' — no valid mode found.")

    # Print summary
    if filled_info:
        print("✅Missing values handled:\n" + "\n".join(filled_info))
    else:
        print("❌No missing values found.")

    return df



def handle_date_features(df):
    df = df.copy()
    date_like_columns = []

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Try parsing with known consistent format first
                converted = pd.to_datetime(df[col], format="%d/%m/%Y", errors='raise')
                df[col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors='coerce')
                date_like_columns.append(col)
              
            except Exception:
                continue  # Not a consistently date-formatted column

    if not date_like_columns:
        print("ℹ️ No date-like columns were found and converted.")
    else:
        print(f"\n📅 Detected and converted date columns: {date_like_columns}")
    
    # print("\n📄 Preview of dataset after date conversion:")
    return df




def encoding_features(df: pd.DataFrame, max_unique_threshold=50) -> pd.DataFrame:
    df = df.copy()

    # Drop irrelevant columns
    drop_cols = ['ID', 'Name', 'Room Number']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Label encode binary categorical column: Gender
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
        print("✅ Label encoded 'Gender'.")

    # Label encode target: Test Results
    if 'Test Results' in df.columns:
        df['Test Results'] = df['Test Results'].map({
            'Normal': 0, 'Abnormal': 1, 'Inconclusive': 2
        })
        print("🎯 Label encoded target column 'Test Results'.")

    # Detect remaining categorical columns
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['Test Results']]

    # Separate based on cardinality
    low_cardinality_cols = [col for col in categorical_cols if df[col].nunique() <= max_unique_threshold]
    high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > max_unique_threshold]

    # One-hot encode low-cardinality columns
    if low_cardinality_cols:
        df = pd.get_dummies(df, columns=low_cardinality_cols, drop_first=True)
        for col in low_cardinality_cols:
            print(f"✅ One-hot encoded '{col}'.")

    # Frequency encode high-cardinality columns
    for col in high_cardinality_cols:
        freq_map = df[col].value_counts()
        df[col] = df[col].map(freq_map)
        print(f"✅ Frequency encoded '{col}'.")

  
    print(f"\n📐 Encoded shape: {df.shape}")
    # print("\n📄 Preview of encoded dataset:")

    return df


def scale_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Define known numerical features
    numeric_cols = ['Age', 'Billing Amount']

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print(f"✅ Scaled numerical columns: {numeric_cols}")
    
    print(f"\n📐 Scaled shape: {df.shape}")
    # print(f"\n📄 Preview of scaled dataset:")

    return df


def save_processed_df(df, filename, output_dir="data/processed"):

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the full file path and save
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)

    # Get the absolute path (Windows-style)
    abs_path = os.path.abspath(filepath)

    print(f"✅ Saved processed DataFrame to:\n{abs_path}")

