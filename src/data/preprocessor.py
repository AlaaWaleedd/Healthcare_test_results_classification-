import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np
import pandas as pd


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


def encode_students_dataset(df: pd.DataFrame, max_unique_threshold=50) -> pd.DataFrame:
    df = df.copy()

    # Label encode binary column: Gender
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    # Label encode target: Test Results
    if 'Test Results' in df.columns:
        df['Test Results'] = df['Test Results'].map({
            'Normal': 0, 'Abnormal': 1, 'Inconclusive': 2
        })

    # Detect object (categorical) columns, excluding already handled and ID-like fields
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['Test Results']]

    # Separate columns with reasonable cardinality
    low_cardinality_cols = [col for col in categorical_cols if df[col].nunique() <= max_unique_threshold]
    high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > max_unique_threshold]

    if high_cardinality_cols:
        print("⚠️ Skipping high-cardinality columns from encoding:", high_cardinality_cols)

    # One-hot encode safe columns
    df = pd.get_dummies(df, columns=low_cardinality_cols, drop_first=True)

    return df







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
    
    print("\n📄 Preview of dataset after date conversion:")
    return df


from sklearn.preprocessing import LabelEncoder

def encode_features(df):
  

    # Drop irrelevant columns
    drop_cols = ['ID', 'Name', 'Room Number']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Label Encoding for binary categorical
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
        print("✅ Label encoded 'Gender'.")

    # One-Hot Encoding for low-cardinality categorical features
    one_hot_cols = ['Blood Type', 'Medical Condition', 'Insurance Provider', 'Admission Type', 'Medication']
    for col in one_hot_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col.replace(" ", "_"))
            print(f"✅ One-hot encoded '{col}'.")

    # Frequency Encoding for high-cardinality features
    freq_encode_cols = ['Doctor', 'Hospital']
    for col in freq_encode_cols:
        if col in df.columns:
            freq_map = df[col].value_counts().to_dict()
            df[col] = df[col].map(freq_map)
            print(f"✅ Frequency encoded '{col}'.")

    # Label Encode the target
    if 'Test Results' in df.columns:
        le = LabelEncoder()
        df['Test Results'] = le.fit_transform(df['Test Results'])
        print(f"🎯 Label encoded target column 'Test Results'.")

    print("\n📄 Preview of encoded dataset:")
    return df.head()



