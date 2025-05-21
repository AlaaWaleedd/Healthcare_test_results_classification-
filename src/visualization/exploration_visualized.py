import matplotlib.dates as mdates
import pandas as pd
from scipy.stats import skew
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_and_plot_missing_values(df):
    missing_data = df.isnull().sum()
    total_rows = df.shape[0]
    missing_summary = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Values': missing_data.values,
        'Missing Percentage': (missing_data / total_rows) * 100
    })
    # Keep only columns with missing values
    missing_summary = missing_summary[missing_summary['Missing Values'] > 0]
    missing_summary = missing_summary.sort_values(by='Missing Percentage', ascending=False)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(missing_summary['Column'], missing_summary['Missing Percentage'], color='coral', alpha=0.7)
    
    plt.title('Missing Data Percentage by Column', fontsize=16)
    plt.ylabel('Percentage of Missing Data (%)', fontsize=12)
    plt.xlabel('Columns', fontsize=12)
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.2f}%', 
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_negative_values(df):
    # Count negative values per column (only numeric columns)
    negative_counts = {}
    for col in df.select_dtypes(include=['number']).columns:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            negative_counts[col] = neg_count
    
    if not negative_counts:
        print("No negative values found in numeric columns.")
        return
    
    # Prepare DataFrame for plotting
    neg_df = pd.DataFrame({
        'Column': list(negative_counts.keys()),
        'Negative Count': list(negative_counts.values())
    }).sort_values(by='Negative Count', ascending=False)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(neg_df['Column'], neg_df['Negative Count'], color='steelblue', alpha=0.7)
    
    plt.title('Count of Negative Values per Column', fontsize=16)
    plt.ylabel('Number of Negative Values', fontsize=12)
    plt.xlabel('Columns', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Label bars with counts
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{int(height)}',
                 ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()



def plot_outliers_all_boxplot(df, threshold=1.5):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("No numeric columns found.")
        return

    n_cols = 3  # Number of plots per row
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()  # flatten in case of multiple rows

    outliers_found = False

    for i, col in enumerate(numeric_cols):
        data = df[col].dropna()

        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Detect if outliers exist
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        if outlier_mask.any():
            outliers_found = True

        axes[i].boxplot(data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='blue'),
                        medianprops=dict(color='red'),
                        flierprops=dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none'))
        axes[i].set_title(f"{col} (Outliers: {outlier_mask.sum()})")
        axes[i].set_ylabel(col)

    # Remove any empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Outlier Detection Using Boxplots (IQR Method)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    if not outliers_found:
        print("❌ No outliers found in any numeric column.")



def plot_distribution_analysis(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        print("⚠️ No numeric columns found in the dataset.")
        return

    for col in numeric_cols:
        data = df[col].dropna()
        skewness = skew(data)

        # Interpret skewness
        if abs(skewness) < 0.5:
            skew_type = "Approximately Normal"
        elif skewness > 0:
            skew_type = "Right-Skewed (Positive)"
        else:
            skew_type = "Left-Skewed (Negative)"

        plt.figure(figsize=(10, 6))
        
        # Histogram + KDE
        sns.histplot(data, bins=40, kde=True, color='skyblue', edgecolor='black', stat='density', alpha=0.7)

        # Rug plot
        sns.rugplot(data, color='black')

        plt.title(f"Distribution of '{col}' | Skewness: {skewness:.2f} → {skew_type}", fontsize=14)
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()



def plot_admissions_heatmap(df):
    # Ensure the date column is datetime
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
    df = df.dropna(subset=['Date of Admission'])

    # Extract Month and Day of Week
    df['Month'] = df['Date of Admission'].dt.strftime('%b')  # e.g., Jan, Feb
    df['DayOfWeek'] = df['Date of Admission'].dt.day_name()  # e.g., Monday

    # Group by Month and Day of Week
    heatmap_data = df.groupby(['Month', 'DayOfWeek']).size().unstack().fillna(0)

    # Reorder for better visual clarity
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(index=months_order, columns=days_order)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='YlGnBu', linewidths=0.5)
    plt.title('Monthly Admissions by Day of Week', fontsize=14)
    plt.xlabel('Day of Week')
    plt.ylabel('Month')
    plt.tight_layout()
    plt.show()


def plot_categorical_by_target(df, categorical_cols, target_col='Test Results'):
    result_colors = {
        'Normal': '#a569bd',
        'Inconclusive': '#85c1e9',
        'Abnormal': '#a3e4d7'
    }

    for col in categorical_cols:
        plt.figure(figsize=(10, 6))

        categories = df[col].dropna().unique()
        targets = df[target_col].dropna().unique()
        bar_width = 0.2
        x = np.arange(len(categories))

        for i, target in enumerate(targets):
            counts = df[df[target_col] == target][col].value_counts().reindex(categories, fill_value=0)
            plt.bar(x + i * bar_width, counts.values, width=bar_width, label=target, color=result_colors.get(target, None))

        plt.xticks(x + bar_width * (len(targets)-1)/2, categories, rotation=45)
        plt.title(f'{col} Distribution by {target_col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.legend(title=target_col)
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()


def plot_target_distribution(df, target_col='Test Results'):
    """Plot distribution of target variable"""
    plt.figure(figsize=(8,6))
    ax = sns.countplot(x=target_col, data=df)
    plt.title('Distribution of Test Results')
    plt.xlabel('Test Result Category')
    plt.ylabel('Count')
    
    # Add percentage labels
    total = len(df[target_col])
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width()/2
        y = p.get_height() + 0.02*total
        ax.annotate(percentage, (x, y), ha='center')
    
    plt.show()
    
    
def plot_numerical_distributions(df, numerical_cols):
    """Plot distributions of numerical features"""
    for col in numerical_cols:
        plt.figure(figsize=(10,4))
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        
        
        plt.tight_layout()
        plt.show()
        
        
import matplotlib.pyplot as plt
import seaborn as sns

def plot_numerical_vs_target_scatter(df, numerical_cols, target_col='Test Results'):
    """Plot numerical features vs target categories with scatter (strip) plots."""
    for col in numerical_cols:
        plt.figure(figsize=(8, 5))
        
        sns.stripplot(
            x=target_col,
            y=col,
            data=df,
            jitter=True,
            hue=target_col,
            palette='Set2',
            size=5,
            dodge=False,
            legend=False
        )
        
        plt.title(f'Distribution of {col} by {target_col}')
        plt.ylabel(col)
        plt.xlabel(target_col)
        
        plt.tight_layout()
        plt.show()
        
        
def plot_monthly_test_result_trends(grouped_df):
    # Ensure the data is sorted by month (if Month is a datetime or categorical)
    if not grouped_df['Month'].dtype.kind in {'i', 'f'}:
        grouped_df = grouped_df.sort_values(by='Month')

    # Get unique test result categories
    test_results = grouped_df['Test Results'].unique()

    # Define a color map similar to Plotly's Set2
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']
    
    plt.figure(figsize=(12, 6))

    for i, result in enumerate(test_results):
        subset = grouped_df[grouped_df['Test Results'] == result]
        plt.plot(
            subset['Month'],
            subset['Count'],
            marker='o',
            label=result,
            color=colors[i % len(colors)]
        )

    plt.title('Monthly Test Result Trends', fontsize=16, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Number of Cases')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Test Results')
    plt.tight_layout()
    plt.show()




def plot_test_result_distribution(df, target_col='Test Results'):
    # Define the custom order
    category_order = ["Normal", "Abnormal", "Inconclusive"]
    
    # Count the frequency of each test result based on the order
    counts = df[target_col].value_counts().reindex(category_order, fill_value=0)

    # Plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values, color=['#88CCEE', '#CC6677', '#DDCC77'])

    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{int(height)}',
                 ha='center', va='bottom', fontsize=10)

    # Styling
    plt.title('Distribution of Test Results', fontsize=14, fontweight='bold')
    plt.xlabel('Test Result')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



def plot_correlation_matrix(df, features):
    # Compute correlation matrix
    corr_matrix = df[features].corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))

    # Draw the heatmap with annotations
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="RdBu_r", 
        center=0,
        square=True, 
        linewidths=0.5, 
        cbar_kws={"shrink": 0.75}
    )

    plt.title('Numerical Features Correlation Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_duplicates(df):
    # Count duplicated rows (excluding the first occurrence)
    duplicated_rows = df.duplicated(keep='first').sum()
    
    # Count duplicated values per column (count values appearing more than once)
    duplicate_counts = {}
    for col in df.columns:
        counts = df[col].value_counts()
        duplicate_counts[col] = counts[counts > 1].sum()  # sum of duplicated values count
    
  

    # Plot duplicated values per column
    plt.figure(figsize=(12, 6))
    plt.bar(duplicate_counts.keys(), duplicate_counts.values(), color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title('Count of Duplicate Values per Column')
    plt.ylabel('Duplicate Count')
    plt.tight_layout()
    plt.show()