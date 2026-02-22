import pandas as pd

WINDOW_MINUTES = 1

def build_windows(csv_path):
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["window"] = df["timestamp"].dt.floor(f"{WINDOW_MINUTES}min")

    windows = []

    for window, group in df.groupby("window"):
        row = {"window": window}

        for signal in group["signal_id"].unique():
            vals = group[group["signal_id"] == signal]["value"]
            row[f"{signal}_mean"] = vals.mean()
            row[f"{signal}_std"] = vals.std() if len(vals) > 1 else 0
            row[f"{signal}_last"] = vals.iloc[-1]

        windows.append(row)

    # Use forward fill then backward fill to handle asynchronous data more robustly than fillna(0)
    return pd.DataFrame(windows).ffill().bfill().fillna(0)