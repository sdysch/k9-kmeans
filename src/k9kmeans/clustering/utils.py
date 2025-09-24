# python
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import pandas as pd

# local
from k9kmeans.logging import setup_logger
from k9kmeans.clustering.preprocessing import load_embeddings


logger = setup_logger(__name__)


def save_clustering_results(
    cluster_df: pd.DataFrame,
    model_type: str,
    embeddings_file: str,
    out_dir: Path,
    cluster_metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save clustering results and metadata to the specified output directory.
    Args:
        cluster_df: pd.DataFrame with clustering results
        model_type: str, type of clustering model (e.g., 'KMeans', 'DBSCAN')
        embeddings_file: str, path to the embeddings file used
        out_dir: Path, directory to save results and metadata
        cluster_metadata: dict, additional metadata about the clustering model
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # save results
    result_file = out_dir / 'cluster_results.csv'
    cluster_df.to_csv(result_file, index=False)

    # save metadata
    meta = {
        'model_type': model_type,
        'timestamp': datetime.now().isoformat(),
        'embeddings_file': str(Path(embeddings_file).resolve()),
        'cluster_metadata': cluster_metadata,
    }
    meta_file = out_dir / 'params.json'
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    logger.debug(f'Saving model metadata: {meta} to {meta_file}')

    logger.info(f'Saved results to {result_file}')
    logger.info(f'Saved metadata to {meta_file}')


def get_clusters_and_embeddings(
    experiment_dir_s: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load clustering results and embeddings from the specified experiment directory.
    Args:
        experiment_dir: str, path to the experiment directory
    Returns:
        df_clusters: pd.DataFrame with clustering results
        df_embeddings: pd.DataFrame with embeddings and filenames
    """
    logger.info(f'Reading experiments from {experiment_dir_s}')
    experiment_dir = Path(experiment_dir_s)
    if not experiment_dir.exists():
        raise ValueError(f'Experiment directory {experiment_dir} does not exist')

    # load clustering results
    clustering_results_file = experiment_dir / 'cluster_results.csv'
    if not clustering_results_file.exists():
        raise ValueError(
            f'Clustering results file {clustering_results_file} does not exist'
        )
    df_clusters = pd.read_csv(clustering_results_file)
    logger.info(
        f'Loaded clustering results from {clustering_results_file} with shape {df_clusters.shape}'
    )

    # load experiment config
    experiment_config_file = experiment_dir / 'params.json'
    if not experiment_config_file.exists():
        raise ValueError(
            f'Experiment config file {experiment_config_file} does not exist'
        )
    with open(experiment_config_file, 'r') as f:
        experiment_config = json.load(f)
    logger.info(f'Experiment config: {experiment_config}')

    # load embeddings
    embeddings_file = experiment_config['embeddings_file']
    df_embeddings = load_embeddings(embeddings_file)

    # check that embeddings and clustering results have the same number of samples before continuing
    if df_embeddings.shape[0] != df_clusters.shape[0]:
        raise ValueError(
            f'Number of samples in embeddings ({df_embeddings.shape[0]}) does not match number of samples in clustering results ({df_clusters.shape[0]})'
        )

    return df_clusters, df_embeddings
