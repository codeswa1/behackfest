from sklearn.cluster import KMeans
import numpy as np

def cluster_events(events):
    if len(events) < 2:
        return events

    X = np.array([[e["severity"]] for e in events])
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X)

    for event, label in zip(events, labels):
        event["cluster"] = int(label)

    return events