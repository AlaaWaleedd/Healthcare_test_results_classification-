import plotly.express as px
import pandas as pd
from scipy.stats import skew
import numpy as np


# Define the function to analyze and plot missing values
def analyze_and_plot_missing_values(df):
    missing_data = df.isnull().sum()
    total_rows = df.shape[0]
    missing_summary = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Values': missing_data.values,
        'Missing Percentage (%)': (missing_data / total_rows) * 100
    })
    # Filter columns with missing values only
    missing_summary = missing_summary[missing_summary['Missing Values'] > 0]
    missing_summary = missing_summary.sort_values(by='Missing Percentage (%)', ascending=False)
    
    # Plot
    fig = px.bar(
        missing_summary,
        x='Column',
        y='Missing Percentage (%)',
        title='Missing Data Percentage by Column',
        text='Missing Percentage (%)',
        labels={'Missing Percentage (%)': '% Missing'},
        color='Missing Percentage (%)',
        color_continuous_scale='OrRd'
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(yaxis_range=[0, 100], xaxis_title='Column Name', yaxis_title='% of Missing Data')
    return fig


def plot_outliers_all(df, threshold=1.5):
   
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    found_outliers = False

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        if outlier_mask.any():
            found_outliers = True
            df_temp = df.copy()
            df_temp['Outlier'] = outlier_mask

            fig = px.scatter(df_temp, x=df_temp.index, y=col, color='Outlier',
                             title=f"Outlier Detection in '{col}' using IQR",
                             labels={'x': 'Index', col: col},
                             color_discrete_map={False: 'blue', True: 'red'})
            fig.update_traces(marker=dict(size=8))
            fig.show()

    if not found_outliers:
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

        # Plot
        fig = px.histogram(data, x=col, nbins=40, marginal="rug", opacity=0.75,
                           title=f"Distribution of '{col}' | Skewness: {skewness:.2f} → {skew_type}")
        fig.update_layout(bargap=0.1)
        fig.show()


def plot_categorical_by_target(df, categorical_cols, target_col='Test Results'):
  
    result_colors = {
        'Normal': '#a569bd',      
        'Inconclusive': '#85c1e9',
        'Abnormal': '#a3e4d7'
    }

    figures = []

    for col in categorical_cols:
        fig = px.histogram(
            df,
            x=col,
            color=target_col,
            color_discrete_map=result_colors,
            barmode='group',
            title=f'<b>{col} Distribution by {target_col}</b>',
            hover_data={col: True, 'Age': True},
            text_auto=True,
            width=800
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            hovermode="x unified",
            bargap=0.25,
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial", size=12),
            title_x=0.5
        )

        figures.append(fig)

    return figures


def plot_monthly_test_result_trends(grouped_df):

    fig = px.line(
        grouped_df,
        x='Month', 
        y='Count', 
        color='Test Results',
        title='<b>Monthly Test Result Trends</b>',
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Number of Cases',
        hovermode='x unified',
        xaxis=dict(tickangle=45),
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig




def plot_test_result_distribution(df, target_col='Test Results'):

    category_order = ["Normal", "Abnormal", "Inconclusive"]
    fig = px.histogram(
        df,
        x=target_col,
        color=target_col,
        title='Distribution of Test Results',
        category_orders={target_col: category_order},
        color_discrete_sequence=px.colors.qualitative.Safe
    )

    fig.update_layout(
        xaxis_title='Test Result',
        yaxis_title='Count',
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )

    return fig



def plot_correlation_matrix(df, features):
   
    corr_matrix = df[features].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu',
        title='Numerical Features Correlation Matrix'
    )

    fig.update_layout(width=600, height=600, title_x=0.5)

  

    return fig

