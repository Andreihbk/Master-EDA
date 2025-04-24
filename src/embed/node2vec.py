# src/embed/node2vec.py
import numpy as np
from node2vec import Node2Vec

def generate_node2vec_embeddings(
    graph,
    dimensions=64,
    walk_length=30,
    num_walks=200,
    window=10,
    min_count=1,
    batch_words=4,
    p=1.0,
    q=1.0,
    seed=42
):
    """
    Run Node2Vec on a NetworkX graph and return the labels & embedding vectors.
    """
    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p, q=q,
        workers=4,
        seed=seed
    )
    model = node2vec.fit(
        window=window,
        min_count=min_count,
        batch_words=batch_words
    )
    labels = model.wv.index_to_key
    vectors = np.array([model.wv[n] for n in labels])
    return labels, vectors

