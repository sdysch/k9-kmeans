# python
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# local
from k9kmeans.logging import setup_logger
from k9kmeans.clustering.utils import save_clustering_results
from k9kmeans.clustering.preprocessing import preprocess_embeddings, load_embeddings
from k9kmeans.clustering.results import ClusterResult


logger = setup_logger(__name__)


def run_dbscan(
    embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = 'euclidean',
    n_jobs: int = -1,
) -> ClusterResult:
    """
    Run DBSCAN clustering on embeddings.
    Args:
        embeddings: np.ndarray of shape (n_samples, n_features)
        eps: float, DBSCAN eps parameter
        min_samples: int, DBSCAN min_samples parameter
        metric: str, distance metric to use
        n_jobs: int, number of parallel jobs to run (-1 means use all available cores
    Returns:
        cluster_labels: np.ndarray of shape (n_samples,) with cluster labels for each sample
    """
    logger.info(
        f'Running DBSCAN with eps={eps}, min_samples={min_samples}, metric={metric}, n_jobs={n_jobs}'
    )
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=n_jobs)
    cluster_labels = dbscan.fit_predict(embeddings)

    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise = list(cluster_labels).count(-1)

    mask = cluster_labels != -1
    try:
        sil_score = silhouette_score(embeddings[mask], cluster_labels[mask])
    except ValueError:
        sil_score = np.nan

    logger.debug(f'Found {num_clusters} clusters')
    logger.debug(f'Found {num_noise} noise points')
    logger.debug(f'silhouette score (excluding noise): {sil_score}')

    return ClusterResult(
        labels=cluster_labels,
        metadata={
            'method': 'dbscan',
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric,
            'num_clusters': num_clusters,
            'num_noise': num_noise,
        },
        silhouette_score=sil_score,
    )


def main(args: argparse.Namespace) -> None:
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(log_level)

    # load embeddings
    logger.info(f'Loading embeddings from {args.embeddings_file}')
    df_embeddings = load_embeddings(args.embeddings_file)
    logger.info(f'Loaded {df_embeddings.shape[0]} embeddings')

    # only need embeddings, not filenames
    X = np.stack(df_embeddings['embedding'].to_numpy())

    # preprocess embeddings
    logger.info('Preprocessing embeddings')
    X = preprocess_embeddings(X)

    # run DBSCAN
    dbscan_params = {
        'eps': args.eps,
        'min_samples': args.min_samples,
        'metric': args.metric,
    }
    dbscan_results = run_dbscan(
        X,
        **dbscan_params,
        n_jobs=args.n_jobs,
    )
    df_clusters = pd.DataFrame(
        {
            'filename': df_embeddings['filename'],
            'cluster': dbscan_results.labels,
        }
    )

    # save results
    logger.info('Saving results')
    save_clustering_results(
        df_clusters,
        model_type='DBSCAN',
        out_dir=Path(args.outdir),
        embeddings_file=args.embeddings_file,
        cluster_metadata=dbscan_results.metadata,
    )
    logger.info('Done')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--outdir',
        type=str,
        required=True,
        help='Output directory that cluster experiments are stored in',
    )
    parser.add_argument(
        '--embeddings_file',
        type=str,
        required=True,
        help='Path to parquet file containing image embeddings',
    )

    # dbscan args
    parser.add_argument(
        '--eps', type=float, default=0.5, help='DBScan eps parameter (default: 0.5)'
    )
    parser.add_argument(
        '--min_samples',
        type=int,
        default=5,
        help='DBScan min_samples parameter (default: 5)',
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='euclidean',
        help='Distance metric (default: euclidean)',
    )
    parser.add_argument(
        '--n_jobs', type=int, default=-1, help='Number of parallel jobs (default: -1)'
    )

    parser.add_argument(
        '-f',
        '--force',
        action='store_true',
        help='If set, overwrite existing output directory',
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    # first check if outdir exists
    outdir = Path(args.outdir)
    if outdir.exists() and not args.force:
        logger.error(
            f'Output directory {outdir} already exists. Use -f/--force to overwrite.'
        )
        exit(1)
    elif outdir.exists() and args.force:
        logger.warning(
            f'Output directory {outdir} already exists and --force set to true. Will overwrite.'
        )

    main(args)
