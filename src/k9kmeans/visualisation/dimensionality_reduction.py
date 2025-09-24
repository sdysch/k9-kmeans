# python
from numpy.typing import NDArray
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca(embeddings: NDArray[np.float64], n_components: int = 2) -> NDArray[np.float64]:
    """
    Perform PCA on the embeddings to reduce their dimensionality.

    Args:
        embeddings: The input embeddings (2D array: n_samples x n_features).
        n_components: The number of principal components to keep.

    Returns:
        The transformed embeddings (2D array: n_samples x n_components).
    """
    pca_model = PCA(n_components=n_components)
    reduced_embeddings: np.ndarray = pca_model.fit_transform(embeddings)
    return reduced_embeddings


def tsne(
    embeddings: NDArray[np.float64], n_components: int = 2, perplexity: float = 30.0
) -> NDArray[np.float64]:
    """
    Perform t-SNE on the embeddings to reduce their dimensionality.

    Args:
        embeddings: The input embeddings (2D array: n_samples x n_features).
        n_components: The number of dimensions to reduce to.
        perplexity: The perplexity parameter for t-SNE.

    Returns:
        The transformed embeddings (2D array: n_samples x n_components).
    """
    tsne_model = TSNE(n_components=n_components, perplexity=perplexity)
    reduced_embeddings: np.ndarray = tsne_model.fit_transform(embeddings)
    return reduced_embeddings
