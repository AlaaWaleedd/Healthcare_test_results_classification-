import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from src.data import preprocessor

def full_pipeline():
    # 📥 Load dataset
    raw_path = "data/raw/train data.csv"
    df = pd.read_csv(raw_path)

    # 🎯 Separate target if it exists
    if 'Test Results' in df.columns:
        target = df['Test Results']
        df = df.drop(columns=['Test Results'])
    else:
        target = None

    # ❌ Drop 'ID' as it's not a feature
    df = df.drop(columns=['ID'], errors='ignore')

    # 🧠 Define feature categories
    numerical_standard = ['Age', 'Room Number']
    numerical_robust = ['Billing Amount']
    categorical = ['Gender', 'Blood Type', 'Insurance Provider']

    # 🛠 Pipelines
    standard_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    robust_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 🔀 Combine all in ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('std_num', standard_pipeline, numerical_standard),
        ('rob_num', robust_pipeline, numerical_robust),
        ('cat', categorical_pipeline, categorical)
    ])

    # 🔁 Transform features
    transformed_array = preprocessor.fit_transform(df)

    # 📐 Create DataFrame
    feature_names = preprocessor.get_feature_names_out()
    df_processed = pd.DataFrame(transformed_array.toarray() if hasattr(transformed_array, 'toarray') else transformed_array,
                                 columns=feature_names)

    # 🎯 Add target back
    if target is not None:
        df_processed['target'] = target.values

    # 💾 Save to file
    os.makedirs("data/processed", exist_ok=True)
    df_processed.to_csv("data/processed/processed_scaled_data.csv", index=False)
    print("✅ Processed and saved to 'data/processed/processed_scaled_data.csv'")

if __name__ == "__main__":
    full_pipeline()
