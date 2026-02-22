import pandas as pd

def build_events(df):
    events = []
    current_event = None

    for _, row in df.iterrows():
        # High-sensitivity: trigger on either behavioral OR structural anomaly
        is_anomaly = row["behavior_anomaly"] or row["structure_anomaly"]

        if is_anomaly:
            if current_event is None:
                current_event = {
                    "start": row["window"],
                    "end": row["window"],
                    "severity": 0.0
                }
            current_event["end"] = row["window"]
            current_event["severity"] += row["behavior_score"]
        else:
            if current_event:
                # Calculate duration in minutes
                delta = pd.to_datetime(current_event["end"]) - pd.to_datetime(current_event["start"])
                current_event["duration"] = max(1, int(delta.total_seconds() / 60))
                events.append(current_event)
                current_event = None

    if current_event:
        delta = pd.to_datetime(current_event["end"]) - pd.to_datetime(current_event["start"])
        current_event["duration"] = max(1, int(delta.total_seconds() / 60))
        events.append(current_event)

    return events