import pandas as pd


def parse_operator_logs(csv_path):
    logs = pd.read_csv(csv_path)
    parsed = []

    for _, row in logs.iterrows():
        text = row["log"].lower()

        if any(word in text for word in ["pressure", "exceeded"]):
            issue = "pressure instability"

        elif any(word in text for word in ["temperature", "thermal"]):
            issue = "thermal response issue"

        elif any(word in text for word in ["vibration", "rpm", "dropped"]):
            issue = "mechanical anomaly"

        elif any(word in text for word in ["cooling", "stabilized", "normal"]):
            issue = "recovery action"

        else:
            issue = "unknown"

        parsed.append({
            "timestamp": row["timestamp"],
            "issue_type": issue
        })

    return parsed