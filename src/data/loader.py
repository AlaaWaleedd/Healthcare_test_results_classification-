import pandas as pd

def load_data(train_path, test_path):
    """
    Load training and test datasets
    
    Args:
        train_path (str): Path to training data CSV
        test_path (str): Path to test data CSV
        
    Returns:
        tuple: (train_df, test_df) pandas DataFrames
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def get_data_overview(df):
    return {
        'info': df.info(),
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'head': df.head()
    }

def classify_features(df, target_col=None):
  
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if target_col and target_col in categorical_cols:
        categorical_cols.remove(target_col)

    print(f"🔸 Found {len(categorical_cols)} categorical features (excluding target '{target_col}'):")
    print(categorical_cols)
    print("\n🔹 Found {0} numerical features:".format(len(numerical_cols)))
    print(numerical_cols)

    return categorical_cols, numerical_cols
