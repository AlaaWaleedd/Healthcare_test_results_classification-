import pandas as pd

from sklearn.preprocessing import StandardScaler

from scipy import stats
import numpy as np

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



def scale_features(df: pd.DataFrame, target_column: str = 'Test Results') -> pd.DataFrame:
    """
    Scales only numeric features using StandardScaler, excluding the target and non-numeric columns.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with numeric and non-numeric features
    target_column : str
        Name of the target column (excluded from scaling)

    Returns:
    --------
    pd.DataFrame
        Scaled DataFrame with target intact, and non-numeric columns untouched
    """
    df = df.copy()

    # Separate target
    y = df[target_column]
    
    # Identify numeric columns (excluding the target)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])

    # Rebuild DataFrame
    scaled_df = df.copy()
    scaled_df[numeric_cols] = X_scaled
    scaled_df[target_column] = y

    return scaled_df

# def encode_categorical_variables(df, verbose=True):
#     # Set display options
#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.width', 1000)
    
#     # Initial report
#     if verbose:
#         print("="*80)
#         print("CATEGORICAL VARIABLE ENCODING VERIFICATION REPORT".center(80))
#         print("="*80)
#         print(f"\nOriginal DataFrame Shape: {df.shape}")
#         print("\nCategorical Columns to encode:")
#         print([col for col in df.columns if df[col].dtype == 'object'])
        
#     # -----------------------------------------------------------------
#     # 1. Admission Type (ordinal encoding)
#     # -----------------------------------------------------------------
#     admission_order = ['Elective', 'Urgent', 'Emergency']
#     df['Admission Type'] = df['Admission Type'].map({val:i for i,val in enumerate(admission_order)})
    
#     if verbose:
#         print("\n" + "="*80)
#         print("ADMISSION TYPE ENCODING VERIFICATION".center(80))
#         print("="*80)
#         print("\nMapping Applied:")
#         print(pd.DataFrame({'Category': admission_order, 'Encoded Value': range(3)}))
#         print("\nValue Distribution After Encoding:")
#         print(df['Admission Type'].value_counts().sort_index())
    
#     # -----------------------------------------------------------------
#     # 2. One-Hot Encoding
#     # -----------------------------------------------------------------
#     nominal_cols = ['Gender', 'Blood Type', 'Medical Condition', 
#                    'Insurance Provider', 'Medication']
    
#     df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    
#     if verbose:
#         print("\n" + "="*80)
#         print("ONE-HOT ENCODING VERIFICATION".center(80))
#         print("="*80)
#         print(f"\nNew Shape: {df_encoded.shape}")
#         print("\nDummy Columns Created:")
#         dummy_report = []
#         for col in nominal_cols:
#             n_dummies = sum(1 for c in df_encoded.columns if c.startswith(col+'_'))
#             dummy_report.append([col, df[col].nunique(), n_dummies])
#         print(pd.DataFrame(dummy_report, 
#                          columns=['Feature', 'Original Categories', 'Dummy Columns']))
    
#     # -----------------------------------------------------------------
#     # 3. Target Encoding Verification
#     # -----------------------------------------------------------------
#     if 'Test Results' in df_encoded.columns:
#         if verbose:
#             print("\n" + "="*80)
#             print("TARGET VARIABLE DISTRIBUTION".center(80))
#             print("="*80)
#             print("\nTest Results Value Counts:")
#             print(df_encoded['Test Results'].value_counts())
#             print("\nClass Balance:")
#             print((df_encoded['Test Results'].value_counts(normalize=True)*100).round(1))
    
#     return df_encoded