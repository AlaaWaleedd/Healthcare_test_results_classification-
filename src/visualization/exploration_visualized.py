import plotly.express as px
import pandas as pd

def plot_missing_data(missing_data: pd.DataFrame):
    """
    Plots a bar chart showing the percentage of missing data per feature.

    Parameters:
        missing_data (pd.DataFrame): A DataFrame with columns 'Missing Values' and 'Percentage (%)',
                                     indexed by feature names.
    """
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
    fig.show()
