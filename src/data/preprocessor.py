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



