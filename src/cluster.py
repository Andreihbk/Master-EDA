# src/cluster.py
from sklearn.cluster import KMeans

def cluster_embeddings(vectors, n_clusters=5, method="kmeans"):
    """
    Cluster embedding vectors and return an array of cluster labels.
    Currently only supports KMeans.
    """
    if method == "kmeans":
        km = KMeans(n_clusters=n_clusters, random_state=42)
        return km.fit_predict(vectors)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
