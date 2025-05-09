import plotly.express as px
import pandas as pd

def plot_missing_data(missing_data: pd.DataFrame):
    
    fig = px.bar(
        missing_data,
        x=missing_data.index,
        y='Percentage (%)',
        text='Missing Values',
        title='Missing Data Percentage by Feature',
        labels={'x': 'Feature', 'Percentage (%)': 'Missing Percentage (%)'},
        color='Percentage (%)',
        color_continuous_scale='Reds'
    )

    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, yaxis_range=[0, 100])
    return fig




def plot_categorical_by_target(df, categorical_cols, target_col='Test Results'):
    """
    Plots grouped bar charts for categorical columns by target classes with custom colors.

    Args:
        df (pd.DataFrame): Input DataFrame.
        categorical_cols (list): Categorical columns to plot.
        target_col (str): Target class column for grouping.

    Returns:
        list: List of Plotly figure objects.
    """
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
    """
    Creates a line plot showing monthly test result trends.

    Args:
        grouped_df (pd.DataFrame): DataFrame with columns ['Month', 'Test Results', 'Count'].

    Returns:
        plotly.graph_objects.Figure: The resulting Plotly figure.
    """
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
    """
    Plots a histogram showing the distribution of test result classes.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The target column representing test results.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure.
    """
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
    """
    Plots and optionally saves a correlation matrix heatmap for selected numerical features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list): List of numerical features to include in the correlation matrix.
        save_path (str, optional): If provided, saves the plot as an SVG at this path.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    corr_matrix = df[features].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu',
        title='Numerical Features Correlation Matrix'
    )

    fig.update_layout(width=600, height=600, title_x=0.5)

  

    return fig
