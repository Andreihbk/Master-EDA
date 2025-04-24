# src/evaluate.py
import pandas as pd
from sklearn.metrics import silhouette_score

def compute_silhouette(vectors, clusters):
    """
    Return the silhouette score for the given embedding vectors and cluster labels.
    """
    return silhouette_score(vectors, clusters)

def precision_at_k(labeled_csv, k=5):
    """
    Compute overall Precision@k from a CSV with columns:
      node, similar_node, similarity, label (0 or 1).
    Assumes exactly k rows per prototype node, and label=1 means “correct”.
    """
    df = pd.read_csv(labeled_csv)
    # For each prototype node, compute the fraction of its top-k similar nodes that are labeled 1
    per_node = df.groupby("node")["label"].mean()
    # Then average across all nodes
    return per_node.mean()
