import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

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


def encode_categorical_variables(df, verbose=True):
    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # Initial report
    if verbose:
        print("="*80)
        print("CATEGORICAL VARIABLE ENCODING VERIFICATION REPORT".center(80))
        print("="*80)
        print(f"\nOriginal DataFrame Shape: {df.shape}")
        print("\nCategorical Columns to encode:")
        print([col for col in df.columns if df[col].dtype == 'object'])
        
    # -----------------------------------------------------------------
    # 1. Admission Type (ordinal encoding)
    # -----------------------------------------------------------------
    admission_order = ['Elective', 'Urgent', 'Emergency']
    df['Admission Type'] = df['Admission Type'].map({val:i for i,val in enumerate(admission_order)})
    
    if verbose:
        print("\n" + "="*80)
        print("ADMISSION TYPE ENCODING VERIFICATION".center(80))
        print("="*80)
        print("\nMapping Applied:")
        print(pd.DataFrame({'Category': admission_order, 'Encoded Value': range(3)}))
        print("\nValue Distribution After Encoding:")
        print(df['Admission Type'].value_counts().sort_index())
    
    # -----------------------------------------------------------------
    # 2. One-Hot Encoding
    # -----------------------------------------------------------------
    nominal_cols = ['Gender', 'Blood Type', 'Medical Condition', 
                   'Insurance Provider', 'Medication']
    
    df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    
    if verbose:
        print("\n" + "="*80)
        print("ONE-HOT ENCODING VERIFICATION".center(80))
        print("="*80)
        print(f"\nNew Shape: {df_encoded.shape}")
        print("\nDummy Columns Created:")
        dummy_report = []
        for col in nominal_cols:
            n_dummies = sum(1 for c in df_encoded.columns if c.startswith(col+'_'))
            dummy_report.append([col, df[col].nunique(), n_dummies])
        print(pd.DataFrame(dummy_report, 
                         columns=['Feature', 'Original Categories', 'Dummy Columns']))
    
    # -----------------------------------------------------------------
    # 3. Target Encoding Verification
    # -----------------------------------------------------------------
    if 'Test Results' in df_encoded.columns:
        if verbose:
            print("\n" + "="*80)
            print("TARGET VARIABLE DISTRIBUTION".center(80))
            print("="*80)
            print("\nTest Results Value Counts:")
            print(df_encoded['Test Results'].value_counts())
            print("\nClass Balance:")
            print((df_encoded['Test Results'].value_counts(normalize=True)*100).round(1))
    
    return df_encoded