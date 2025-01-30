import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statistical_test import graph_ranks
import plotly.express as px
import streamlit as st

FONT_SIZE = 24

def plot_anomaly_figure(anomaly_df, 
                        fig_width=1500, 
                        normal_color='#91ccfa', 
                        abnormal_color='red'):

    # Get the number of time steps
    time_steps = list(range(len(anomaly_df)))

    # 1️st PLOT: Data column with labels highlighted
    fig1 = go.Figure()

    # Plot all data in blue
    fig1.add_trace(go.Scatter(
        x=time_steps, y=anomaly_df["data"], mode="lines", name="Data",
        line=dict(color=normal_color)
    ))

    # Highlight label==1 in red
    fig1.add_trace(go.Scatter(
        x=time_steps, y=anomaly_df["data"].where(anomaly_df["label"] == 1),
        mode="markers", name="Anomalies",
        marker=dict(color=abnormal_color, size=5)
    ))

    fig1.update_layout(
                       title=dict(
                        text="Anomalous Time Series (Data & Labels)",
                        font=dict(size=FONT_SIZE)  # Increase title font size
                        ),
                       xaxis_title="Time",
                       yaxis_title="Anomaly Score",
                       xaxis=dict(title_font=dict(size=FONT_SIZE-4)),  # Increase x-axis label font size
                       yaxis=dict(title_font=dict(size=FONT_SIZE-4)),  # Increase y-axis label font size
                       template="plotly_white",
                       width=fig_width)

    # 2️nd PLOT: Prediction column with a hidden legend for alignment
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=time_steps, y=anomaly_df["prediction"], mode="lines", name="Prediction",
        line=dict(color=normal_color)
    ))

    # Add a dummy invisible trace to match the legend spacing
    fig2.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name=" ",  # Empty legend entry
        marker=dict(color="white")
    ))

    fig2.update_layout(title=dict(
                        text="Anomaly Detection Over Time",
                        font=dict(size=FONT_SIZE)  # Increase title font size
                        ),
                       xaxis_title="Time",
                       yaxis_title="Anomaly Score",
                       xaxis=dict(title_font=dict(size=FONT_SIZE-4)),  # Increase x-axis label font size
                       yaxis=dict(title_font=dict(size=FONT_SIZE-4)),  # Increase y-axis label font size
                       template="plotly_white",
                       width=fig_width,
                       showlegend=True)  # Ensures a consistent legend area

    # Set the same x-axis range for both plots
    fig1.update_xaxes(range=[0, len(anomaly_df)])
    fig2.update_xaxes(range=[0, len(anomaly_df)])

    # Display in Streamlit
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)