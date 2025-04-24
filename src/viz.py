import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_tsne_clusters(embeddings, cluster_labels, title="t-SNE Clusters", figsize=(10,8), save_path=None):
    """
    Scatter-plot a 2D TSNE embedding colored by cluster labels.
    embeddings: np.array of shape (n_points, 2)
    cluster_labels: list/array of ints for each point
    """
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(embeddings[:,0], embeddings[:,1], c=cluster_labels, cmap='tab10', s=30, alpha=0.7)
    cbar = fig.colorbar(scatter, ax=ax, ticks=sorted(set(cluster_labels)))
    cbar.set_label('Cluster')
    ax.set_title(title)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_tsne_types(embeddings, type_labels, title="t-SNE Node Types", figsize=(10,8), save_path=None):
    """
    Scatter-plot a 2D TSNE embedding colored by node-type labels (strings).
    embeddings: np.array of shape (n_points, 2)
    type_labels: list of string labels for each point
    """
    unique_types = sorted(set(type_labels))
    mapping = {t:i for i,t in enumerate(unique_types)}
    codes = [mapping[t] for t in type_labels]
    cmap = plt.cm.get_cmap('tab20', len(unique_types))
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(embeddings[:,0], embeddings[:,1], c=codes, cmap=cmap, s=30, alpha=0.7)
    # Legend
    handles = [mpatches.Patch(color=cmap(i), label=t) for i,t in enumerate(unique_types)]
    ax.legend(handles=handles, loc='best', bbox_to_anchor=(1,1), title="Node Type")
    ax.set_title(title)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax
