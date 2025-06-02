import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import umap
from typing import Union, Optional
from DataManagment import Dataset, SiameseDataset
import seaborn as sns
import pandas as pd


def visualize_2d_projection(
    dataset: Union[Dataset, SiameseDataset],
    title_prefix: str = "",
    sample_size: Optional[int] = None,
    random_seed: int = 42,
    show: bool = True,
    save_dir: Optional[str] = None
):
    """
    Visualize a dataset or Siamese dataset using PCA, t-SNE, and UMAP.

    Parameters:
        dataset: Dataset or SiameseDataset instance
        title_prefix: Prefix to use in plot titles
        sample_size: Number of samples to visualize (None = use all)
        random_seed: Seed for reproducibility
        show: Whether to show plots using plt.show()
        save_dir: If provided, saves plots to this directory
    """
    print(f"Visualizing dataset: {dataset.name if hasattr(dataset, 'name') else 'Unnamed Dataset'}")
    print(f"Sample size: {sample_size if sample_size is not None else 'All samples'}")

    np.random.seed(random_seed)

    if isinstance(dataset, SiameseDataset):
        data = dataset.pairs
        X = np.array([np.concatenate([x1, x2]) for x1, x2, _ in data])
        y = np.array([label for _, _, label in data])
        if not np.issubdtype(y.dtype, np.number):
            y = LabelEncoder().fit_transform(y)
    else:
        X, y = dataset.get_processed_data()
        if not np.issubdtype(y.dtype, np.number):
            y = LabelEncoder().fit_transform(y)
        if sample_size and len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X, y = X[indices], y[indices]

    if not isinstance(dataset, SiameseDataset):
        df = pd.DataFrame(X, columns=dataset.feature_names)
        df['label'] = y
        # For large feature sets, limit to first few columns
        selected_features = dataset.feature_names[:min(5, len(dataset.feature_names))]
        sns.pairplot(df, vars=selected_features, hue='label', palette="viridis", corner=True)
        if show:
            plt.show()


    projections = {
        "PCA": PCA(n_components=2, random_state=random_seed).fit_transform(X),
        "t-SNE": TSNE(n_components=2, random_state=random_seed, perplexity=30, n_iter=1000).fit_transform(X),
        "UMAP": umap.UMAP(n_components=2, random_state=random_seed).fit_transform(X),
    }

    # Plot side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (method, X_2d) in zip(axes, projections.items()):
        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="viridis", alpha=0.7, s=10)
        ax.set_title(f"{title_prefix} - {method}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    fig.colorbar(sc, ax=axes, shrink=0.6, label="Class")
    fig.suptitle(f"2D Projections: {title_prefix}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir:
        path = f"{save_dir}/{title_prefix}_projections.png".replace(" ", "_")
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
