import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler


class AutoEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def detect_behavior_anomalies(df):
    features = df.drop(columns=["window"])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features.values)

    X = torch.tensor(scaled, dtype=torch.float32)

    # Train only on first 20% (strictly normal period in our expanded data)
    # This prevents the model from "learning" the anomalies as normal behavior.
    split_idx = int(0.2 * len(X))
    X_train = X[:split_idx]
    X_test = X

    model = AutoEncoder(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Train only on normal portion
    for epoch in range(500):
        optimizer.zero_grad()
        recon = model(X_train)
        loss = loss_fn(recon, X_train)
        loss.backward()
        optimizer.step()

    # Evaluate on full dataset
    with torch.no_grad():
        recon_full = model(X_test)
        # Use per-sample MSE
        errors = ((X_test - recon_full) ** 2).mean(dim=1).numpy()

    df["behavior_score"] = errors

    # --- Dynamic Thresholding (Inspired by POT) ---
    # Instead of a fixed Mean + 3*Std (which assumes Gaussian noise),
    # we look at the tail of the distribution.
    train_errors = errors[:split_idx]
    mean_train = np.mean(train_errors)
    std_train = np.std(train_errors)

    # 1. Base threshold (Gaussian assumption)
    base_threshold = mean_train + 3 * std_train
    
    # 2. Tail-based threshold (Percentile)
    # We look at where the "peaks" start to emerge.
    # If the distribution is heavy-tailed, the 98th percentile will be
    # significantly higher than the base threshold.
    tail_threshold = np.percentile(errors, 98)
    
    # Selection logic: choose the more conservative one if noise is high,
    # or the more sensitive one if the system is very stable.
    if std_train < 1e-4:
        # High confidence in training data, any deviation is a potential anomaly
        threshold = max(base_threshold, np.percentile(errors, 90))
    else:
        # More noise, look for significant peaks over the threshold
        threshold = max(base_threshold, tail_threshold)

    print(f"Training split index: {split_idx}")
    print(f"Max train error: {train_errors.max():.6f}")
    print(f"98th percentile error: {tail_threshold:.6f}")
    print(f"Final Dynamic Threshold: {threshold:.6f}")

    df["behavior_anomaly"] = df["behavior_score"] > threshold

    return df