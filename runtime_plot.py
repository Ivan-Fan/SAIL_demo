import pandas as pd
import plotly.express as px


def create_plot_varylen(df_accuracy_varylen, df_runtime_varylen, LINE_WIDTH = 5, MARKER_SIZE = 13, FONT_SIZE = 20, SMALL_FONT_SIZE = 13):
    
    # Create Accuracy Plot
    fig_accuracy = px.line(
        df_accuracy_varylen,
        x="Time-series length",
        y="Accuracy",
        color="Method",
        symbol="Method",
        line_dash="Method",
        markers=True,
        log_x=True,  # Log scale for x-axis
    )

    fig_accuracy.update_layout(
        title=dict(
        text="Accuracy vs. Time-series Length",
        font=dict(size=FONT_SIZE+5),
        x=0.5,  # Center the title
        xanchor="center"  # Ensure it's properly aligned
        ),  # Title font size
        xaxis=dict(title=dict(text="Time-series length", font=dict(size=FONT_SIZE)), tickfont=dict(size=FONT_SIZE-2)),
        yaxis=dict(title=dict(text="Accuracy", font=dict(size=FONT_SIZE)), tickfont=dict(size=FONT_SIZE-2)),

        legend=dict(
                title=dict(text=""),
                orientation="h",  # Keep horizontal legend
                yanchor="bottom",  # Anchor legend to the bottom of the figure
                y=0.97,  # Move legend slightly above the plot
                xanchor="center",  # Center-align the legend
                x=0.5,  # Place the legend in the middle
                font=dict(size=FONT_SIZE-3)  # Increase legend font size for clarity
            ),
        margin=dict(l=40, r=40, t=80, b=40)  # Adjust top margin for better spacing
    )

    fig_accuracy.update_traces(line=dict(width=LINE_WIDTH), 
                              marker=dict(size=MARKER_SIZE))

    # Create Runtime Plot
    fig_runtime = px.line(
        df_runtime_varylen,
        x="Time-series length",
        y="Runtime (s)",
        color="Method",
        symbol="Method",
        line_dash="Method",
        markers=True,
        log_x=True,  # Log scale for x-axis
        log_y=True,  # Log scale for y-axis (Runtime in seconds)
    )
    fig_runtime.update_traces(line=dict(width=LINE_WIDTH), 
                              marker=dict(size=MARKER_SIZE))

    fig_runtime.update_layout(
        title=dict(
        text="Runtime vs. Time-series Length",
        font=dict(size=FONT_SIZE+5),
        x=0.5,  # Center the title
        xanchor="center"  # Ensure it's properly aligned
        ),  # Title font size
        xaxis=dict(title=dict(text="Time-series length", font=dict(size=FONT_SIZE)), tickfont=dict(size=FONT_SIZE-2)),
        yaxis=dict(title=dict(text="Runtime (s) (log scale)", font=dict(size=FONT_SIZE)), tickfont=dict(size=FONT_SIZE-2)),

        legend=dict(
                title=dict(text=""),
                orientation="h",  # Keep horizontal legend
                yanchor="bottom",  # Anchor legend to the bottom of the figure
                y=0.97,  # Move legend slightly above the plot
                xanchor="center",  # Center-align the legend
                x=0.5,  # Place the legend in the middle
                font=dict(size=FONT_SIZE-3)  # Increase legend font size for clarity
            ),
        margin=dict(l=40, r=40, t=80, b=40)  # Adjust top margin for better spacing
    )

    return fig_accuracy, fig_runtime


def create_plot_varynum(df_accuracy_varynum, df_runtime_varynum, LINE_WIDTH = 5, MARKER_SIZE = 13, FONT_SIZE = 20, SMALL_FONT_SIZE = 13):
    
    # Create Accuracy Plot
    fig_accuracy = px.line(
        df_accuracy_varynum,
        x="Number of Time-series",
        y="Accuracy",
        color="Method",
        symbol="Method",
        line_dash="Method",
        markers=True,
        log_x=True,  # Log scale for x-axis
    )

    fig_accuracy.update_layout(
        title=dict(
        text="Accuracy vs. Number of Time-series",
        font=dict(size=FONT_SIZE+5),
        x=0.5,  # Center the title
        xanchor="center"  # Ensure it's properly aligned
        ),  # Title font size
        xaxis=dict(title=dict(text="Number of Time-series", font=dict(size=FONT_SIZE)), tickfont=dict(size=FONT_SIZE-2)),
        yaxis=dict(title=dict(text="Accuracy", font=dict(size=FONT_SIZE)), tickfont=dict(size=FONT_SIZE-2)),

        legend=dict(
                title=dict(text=""),
                orientation="h",  # Keep horizontal legend
                yanchor="bottom",  # Anchor legend to the bottom of the figure
                y=0.97,  # Move legend slightly above the plot
                xanchor="center",  # Center-align the legend
                x=0.5,  # Place the legend in the middle
                font=dict(size=FONT_SIZE-3)  # Increase legend font size for clarity
            ),
        margin=dict(l=40, r=40, t=80, b=40)  # Adjust top margin for better spacing
    )

    fig_accuracy.update_traces(line=dict(width=LINE_WIDTH), 
                              marker=dict(size=MARKER_SIZE))

    # Create Runtime Plot
    fig_runtime = px.line(
        df_runtime_varynum,
        x="Number of Time-series",
        y="Runtime (s)",
        color="Method",
        symbol="Method",
        line_dash="Method",
        markers=True,
        log_x=True,  # Log scale for x-axis
        log_y=True,  # Log scale for y-axis (Runtime in seconds)
    )
    fig_runtime.update_traces(line=dict(width=LINE_WIDTH), 
                              marker=dict(size=MARKER_SIZE))

    fig_runtime.update_layout(
        title=dict(
        text="Runtime vs. Number of Time-series",
        font=dict(size=FONT_SIZE+5),
        x=0.5,  # Center the title
        xanchor="center"  # Ensure it's properly aligned
        ),  # Title font size
        xaxis=dict(title=dict(text="Number of Time-series", font=dict(size=FONT_SIZE)), tickfont=dict(size=FONT_SIZE-2)),
        yaxis=dict(title=dict(text="Runtime (s) (log scale)", font=dict(size=FONT_SIZE)), tickfont=dict(size=FONT_SIZE-2)),

        legend=dict(
                title=dict(text=""),
                orientation="h",  # Keep horizontal legend
                yanchor="bottom",  # Anchor legend to the bottom of the figure
                y=0.97,  # Move legend slightly above the plot
                xanchor="center",  # Center-align the legend
                x=0.5,  # Place the legend in the middle
                font=dict(size=FONT_SIZE-3)  # Increase legend font size for clarity
            ),
        margin=dict(l=40, r=40, t=80, b=40)  # Adjust top margin for better spacing
    )

    return fig_accuracy, fig_runtime