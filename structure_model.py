import numpy as np

def detect_structure_anomalies(df, window_size=5):
    signal_cols = [c for c in df.columns if "_mean" in c]
    
    # Calculate rolling correlations (requires at least window_size points)
    # This detects when the RELATIONSHIP between signals breaks suddenly
    scores = []
    
    for i in range(len(df)):
        if i < window_size:
            scores.append(0.0)
            continue
            
        # Current local baseline (last 'window_size' steps)
        history = df.iloc[i-window_size:i][signal_cols]
        baseline_corr = history.corr().fillna(0).values
        
        # Current observation relationship
        # outer product captures the instantaneous relationship
        current_vals = df.iloc[i][signal_cols].values
        current_rel = np.outer(current_vals, current_vals)
        
        # Normalize current_rel to be comparable to correlation if needed, 
        # but here we just look for sudden departures from the local trend.
        # A simpler way is to compare current correlation with rolling correlation
        
        # Improved: Compare current correlation matrix (last 2 samples) with local history
        current_window = df.iloc[max(0, i-1):i+1][signal_cols]
        current_corr = current_window.corr().fillna(0).values
        
        drift = np.abs(current_corr - baseline_corr).mean()
        scores.append(drift)

    df["structure_score"] = scores
    # Dynamic threshold based on the scores in this run
    threshold = np.percentile(scores, 95)
    df["structure_anomaly"] = df["structure_score"] > threshold

    return df