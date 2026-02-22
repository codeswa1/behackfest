import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def plot_raw_timeseries(csv_path):
    """Interactive raw data exploration."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    fig = go.Figure()
    for signal in df["signal_id"].unique():
        subset = df[df["signal_id"] == signal]
        fig.add_trace(go.Scatter(
            x=subset["timestamp"],
            y=subset["value"],
            name=signal,
            mode='markers',
            marker=dict(size=5)
        ))

    fig.update_layout(
        title="Physical Sensor Exploration (Raw Asynchronous Data)",
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="closest"
    )
    return fig


def plot_anomaly_scores(df):
    """Interactive trend analysis for scores."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["window"],
        y=df["behavior_score"],
        name="Behavior Score",
        mode='lines+markers'
    ))
    fig.add_trace(go.Scatter(
        x=df["window"],
        y=df["structure_score"],
        name="Structure Score",
        mode='lines+markers'
    ))

    fig.update_layout(
        title="Anomaly Score Trends over Time",
        xaxis_title="Time Window",
        yaxis_title="Normalized Score",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig


def plot_events_and_logs(windows, events, timeseries_csv, log_csv):
    """Interactive Plotly dashboard for anomalies and logs."""
    logs = pd.read_csv(log_csv)
    logs["timestamp"] = pd.to_datetime(logs["timestamp"])
    
    # Load raw data for context
    raw_df = pd.read_csv(timeseries_csv)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Anomaly Scores (High-Level)", "Raw Sensor Data (Physical)"),
        row_heights=[0.4, 0.6]
    )

    # 1. Plot Anomaly Scores (Top Axis)
    fig.add_trace(
        go.Scatter(x=windows["window"], y=windows["behavior_score"], name="Behavior Score", line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=windows["window"], y=windows["structure_score"], name="Structure Score", line=dict(color='green')),
        row=1, col=1
    )

    # 2. Plot Raw Sensor Data (Bottom Axis)
    signals = raw_df["signal_id"].unique()
    for sig in signals:
        sig_data = raw_df[raw_df["signal_id"] == sig]
        fig.add_trace(
            go.Scatter(
                x=sig_data["timestamp"], 
                y=sig_data["value"], 
                name=f"Sensor: {sig}",
                mode='lines+markers',
                marker=dict(size=4),
                line=dict(width=1),
                opacity=0.7
            ),
            row=2, col=1
        )

    # 3. Add Event Regions (Shaded Spans)
    legend_added = set()
    for i, event in enumerate(events):
        start = pd.to_datetime(event["start"])
        end = pd.to_datetime(event["end"])
        cluster = event.get('cluster', 0)
        color = 'LightCoral' if cluster == 1 else 'navajowhite'
        label = "High Severity Event" if cluster == 1 else "Standard Anomaly"
        
        # Add highlight
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color, opacity=0.3,
            layer="below", line_width=0,
            annotation_text=f" Event {i}", 
            annotation_position="top left",
            row="all", col=1
        )
        
        # Add midpoint marker star
        midpoint = start + (end - start) / 2
        fig.add_trace(
            go.Scatter(
                x=[midpoint], 
                y=[windows["behavior_score"].max() * 1.05],
                mode="markers",
                marker=dict(color="red", size=15, symbol="star"),
                name="Event Peak (Legend Marker)",
                hovertext=f"Event {i} Center<br>Type: {label}",
                showlegend=True if "peak" not in legend_added else False
            ),
            row=1, col=1
        )
        legend_added.add("peak")

    # 4. Add Operator Logs (Interactive Lines)
    for i, (_, row) in enumerate(logs.iterrows()):
        log_ts = pd.to_datetime(row["timestamp"])
        log_text = row["log"]
        
        fig.add_vline(
            x=log_ts,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            row="all", col=1
        )
        
        # Invisible trace for hover tooltip
        fig.add_trace(
            go.Scatter(
                x=[log_ts, log_ts],
                y=[0, windows["behavior_score"].max() * 1.2],
                mode="lines",
                line=dict(width=0),
                name="Operator Chat Log",
                hoverinfo="text",
                hovertext=f"<b>Operator Log:</b><br>{log_text}",
                showlegend=True if i == 0 else False
            ),
            row=1, col=1
        )

    # Layout Polish
    fig.update_layout(
        height=850,
        title_text="Anomaly Detection Dashboard (Interactive)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis2_rangeslider_visible=True,
    )

    fig.update_yaxes(title_text="Anomaly Score", row=1, col=1)
    fig.update_yaxes(title_text="Sensor Value", row=2, col=1)

    # Save outputs
    fig.write_html("anomaly_dashboard.html")
    # Try to save static image
    try:
        import kaleido
        fig.write_image("anomaly_events_plot.png")
        print("\n[Plotting] Static preview updated at anomaly_events_plot.png")
    except ImportError:
        pass

    print(f"[Plotting] INTERACTIVE DASHBOARD READY: anomaly_dashboard.html")
    return fig
