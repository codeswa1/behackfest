from preprocessing import build_windows
from anomaly_model import detect_behavior_anomalies
from structure_model import detect_structure_anomalies
from event_builder import build_events
from clustering import cluster_events
from llm_parser import parse_operator_logs

from plotting import (
    plot_raw_timeseries,
    plot_anomaly_scores,
    plot_events_and_logs
)

print("\n==============================")
print(" ANOMALY DETECTION DEMO START ")
print("==============================")

# 1. Plot raw async data (problem illustration)
print("\n[1] Plotting raw asynchronous sensor data...")
plot_raw_timeseries("data/timeseries.csv")

# 2. Preprocess async data → windows
print("\n[2] Building time windows...")
windows = build_windows("data/timeseries.csv")

# 3. Behavioral anomaly detection
print("\n[3] Detecting behavioral anomalies...")
windows = detect_behavior_anomalies(windows)

# 4. Structural anomaly detection
print("\n[4] Detecting structural anomalies...")
windows = detect_structure_anomalies(windows)

# 5. Plot anomaly scores
print("\n[5] Plotting anomaly scores...")
plot_anomaly_scores(windows)

# 6. Build anomalous events
print("\n[6] Building anomalous events...")
events = build_events(windows)

# 7. Cluster events
print("\n[7] Clustering events...")
events = cluster_events(events)

# 8. Parse operator logs
print("\n[8] Parsing operator logs...")
parsed_logs = parse_operator_logs("data/operator_logs.csv")

# 9. Plot detected events vs operator logs
print("\n[9] Plotting events aligned with operator logs...")
plot_events_and_logs(
    windows,
    events,
    "data/timeseries.csv",
    "data/operator_logs.csv"
)

# 10. Print summary (for console demo)
print("\n========== DETECTED EVENTS ==========")
for e in events:
    print(e)

print("\n========== OPERATOR LOG ISSUES ==========")
for log in parsed_logs:
    print(log)

print("\n==============================")
print(" DEMO COMPLETED SUCCESSFULLY ")
print("==============================")