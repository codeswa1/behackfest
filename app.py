import streamlit as st
import pandas as pd
import os
from preprocessing import build_windows
from anomaly_model import detect_behavior_anomalies
from structure_model import detect_structure_anomalies
from event_builder import build_events
from clustering import cluster_events
from llm_parser import parse_operator_logs
from plotting import plot_events_and_logs, plot_raw_timeseries, plot_anomaly_scores

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")

st.title("Anomaly Detection & Diagnostic Dashboard")
st.markdown("""
Upload your sensor data and operator logs to detect anomalies and correlate them with human observations.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")
uploaded_ts = st.sidebar.file_uploader("Upload Time-Series CSV", type=["csv"])
uploaded_logs = st.sidebar.file_uploader("Upload Operator Logs CSV", type=["csv"])

# AI Diagnostic Configuration
st.sidebar.divider()
st.sidebar.header("AI Diagnostic Settings")
ai_mode = st.sidebar.radio(
    "AI Provider",
    ["Groq (Free)", "Gemini (Free)", "OpenAI", "Ollama (Local)"],
    help="Groq and Gemini have free tiers. Ollama works locally only."
)

if ai_mode == "Groq (Free)":
    st.sidebar.markdown("**Free** · Get key at [console.groq.com](https://console.groq.com)")
    api_key = st.sidebar.text_input("Groq API Key", type="password")
    ai_provider = "groq"
elif ai_mode == "Gemini (Free)":
    st.sidebar.markdown("**Free** · Get key at [aistudio.google.com](https://aistudio.google.com)")
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    ai_provider = "gemini"
elif ai_mode == "OpenAI":
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    ai_provider = "openai"
else:
    st.sidebar.info("Using local **Ollama** with `llama3.2:latest`. Does not work on cloud.")
    api_key = "local_ollama"
    ai_provider = "ollama"

if uploaded_ts and uploaded_logs:
    # Save temporary files
    with open("temp_ts.csv", "wb") as f:
        f.write(uploaded_ts.getbuffer())
    with open("temp_logs.csv", "wb") as f:
        f.write(uploaded_logs.getbuffer())

    st.success("Files uploaded successfully! Processing...")

    # Pipeline execution
    with st.spinner("Running Detection Pipeline..."):
        # 1. Preprocess
        windows = build_windows("temp_ts.csv")
        
        # 2. behavioral
        windows = detect_behavior_anomalies(windows)
        
        # 3. structural
        windows = detect_structure_anomalies(windows)
        
        # 4. Events
        events = build_events(windows)
        
        # 5. Clustering
        events = cluster_events(events)
        
        # 6. Parse Logs
        parsed_logs = parse_operator_logs("temp_logs.csv")
        df_logs = pd.read_csv("temp_logs.csv") # Define globally for the UI block

    st.header(" Analysis Results")
    
    # 0. Insight Metrics (High Level)
    m1, m2, m3 = st.columns(3)
    total_drop = len(events) * 15
    health_score = max(0, 100 - total_drop)
    
    m1.metric(
        "System Health Score", 
        f"{health_score}%", 
        delta=f"-{total_drop}%" if events else "Stable",
        delta_color="inverse"
    )
    m2.metric("Total Events Detected", len(events))
    high_sev = len([e for e in events if e.get('cluster', 0) == 1])
    m3.metric("High Severity Alerts", high_sev, delta_color="inverse")

    # 1. Visualization Tabs
    tab1, tab2, tab3 = st.tabs(["Integrated Diagnosis", "Raw Signal Explorer", "Anomaly Score Trends"])

    with tab1:
        st.subheader("Interactive Event Dashboard")
        fig_events = plot_events_and_logs(windows, events, "temp_ts.csv", "temp_logs.csv")
        st.plotly_chart(fig_events, width="stretch")
    
    with tab2:
        st.subheader("Raw Asynchronous Sensor Data")
        fig_raw = plot_raw_timeseries("temp_ts.csv")
        st.plotly_chart(fig_raw, width="stretch")
        
    with tab3:
        st.subheader("Behavioral & Structural Scores")
        fig_scores = plot_anomaly_scores(windows)
        st.plotly_chart(fig_scores, width="stretch")

    # 2. Event Summary Tables
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detected Events")
        if events:
            df_events = pd.DataFrame(events)
            st.dataframe(df_events[["start", "end", "severity", "cluster"]])
        else:
            st.info("No anomalies detected.")

    with col2:
        st.subheader("Operator Logs")
        st.dataframe(df_logs)

    # 3. AI Diagnostic Reasoning (Advanced Layer - Placed last to prevent blocking charts)
    st.divider()
    st.header(" AI Diagnostic Reasoning")
    if events:
        for i, event in enumerate(events):
            with st.expander(f"Detailed Analysis: Event {i}", expanded=(i==0)):
                e_start = pd.to_datetime(event["start"])
                e_end = pd.to_datetime(event["end"])
                
                related_logs = df_logs[
                    (pd.to_datetime(df_logs["timestamp"]) >= e_start - pd.Timedelta(minutes=5)) &
                    (pd.to_datetime(df_logs["timestamp"]) <= e_end + pd.Timedelta(minutes=5))
                ].to_dict('records')
                
                context = {
                    "severity": "CRITICAL" if event.get('cluster', 0) == 1 else "WARNING",
                    "behavior_score": f"{event.get('severity', 0):.4f}",
                    "duration": event.get('duration', 'unknown'),
                    "logs": related_logs
                }

                # 1. Standard Rule-Based Diagnostic
                st.markdown(f"#### Standard Diagnostic Analysis")
                st.markdown(f"**Anomaly Type**: `{context['severity']}` Behavioral Drift")
                
                reasoning = f"""
                **Reasoning Engine Output**:
                - The system detected a deviation in the **AutoEncoder reconstruction path** (Score: {context['behavior_score']}).
                - Event Duration: **{context['duration']} minutes**.
                """
                
                if related_logs:
                    log_summary = " ".join([l['log'] for l in related_logs])
                    root_cause = f"The correlation with log entries (*'{log_summary}'*) suggests an external intervention or a known subsystem failure triggered this event."
                else:
                    root_cause = "No matching operator logs found. This suggests a **Silent Failure** or an internal relationship drift that was not immediately visible to human operators."
                
                st.write(reasoning)
                st.success(f"**Final Assessment**: {root_cause}")

                # 2. Advanced AI Reasoning Analysis (Optional/Live)
                if api_key:
                    st.divider()
                    st.markdown(f"####  Advanced AI Reasoning ({ai_mode})")
                    with st.spinner(f"Generating Analysis for Event {i}..."):
                        from llm_service import get_ai_diagnosis
                        real_diagnosis = get_ai_diagnosis(api_key, context, provider=ai_provider)
                        st.markdown(real_diagnosis)
    else:
        st.success("System is operating within normal behavioral parameters. No active insights required.")

else:
    st.info("Please upload both Time-Series Data and Operator Logs in the sidebar to begin.")
    
    # Provide sample data download links or instructions
    st.subheader("Don't have data?")
    st.markdown("""
    You can use the `generate_data.py` script to create sample files in the `data/` directory.
    - `data/timeseries.csv`
    - `data/operator_logs.csv`
    """)

st.divider()
st.caption("Advanced Anomaly Detection System v1.0 | Powered by AutoEncoders & Plotly")
