# python
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from typing import cast
import numpy as np
import pandas as pd

# local
from k9kmeans.logging import setup_logger


logger = setup_logger(__name__)


def preprocess_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Preprocess embeddings by scaling them to have zero mean and unit variance.
    Args:
        embeddings: np.ndarray of shape (n_samples, n_features)
    Returns:
        embeddings_scaled: np.ndarray of shape (n_samples, n_features)
    """
    logger.info('Preprocessing embeddings with StandardScaler')
    embeddings_scaled = StandardScaler().fit_transform(embeddings)
    return cast(NDArray[np.float32], embeddings_scaled)


def load_embeddings(file_path: str) -> pd.DataFrame:
    """
    Load embeddings from a Parquet or CSV file.
    The file must contain an 'embedding' column with list-like embeddings.
    Args:
        file_path: str, path to the Parquet or CSV file
    Returns:
        df: pd.DataFrame with an 'embedding' column
    """
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        logger.error('Unsupported file format. Use .parquet or .csv')
        raise ValueError('Unsupported file format. Use .parquet or .csv')
    if 'embedding' not in df.columns:
        logger.error('DataFrame must contain an "embedding" column')
        raise ValueError('DataFrame must contain an "embedding" column')
    return df
