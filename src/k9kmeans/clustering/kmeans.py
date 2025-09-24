# python
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.cluster import KMeans

# local
from k9kmeans.logging import setup_logger
from k9kmeans.clustering.utils import save_clustering_results
from k9kmeans.clustering.preprocessing import preprocess_embeddings, load_embeddings
from k9kmeans.clustering.results import ClusterResult


logger = setup_logger(__name__)


def run_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    max_iter: int = 300,
) -> ClusterResult:
    """
    Run KMeans clustering on embeddings.
    Args:
        embeddings: np.ndarray of shape (n_samples, n_features)
        n_clusters: int, number of clusters to form
        max_iter: int, maximum number of iterations of the k-means algorithm
    Returns:
        cluster_labels: np.ndarray of shape (n_samples,) with cluster labels for each sample
    """
    logger.info(f'Running KMeans with n_clusters={n_clusters}, max_iter={max_iter}')
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter)
    cluster_labels = kmeans.fit_predict(embeddings)

    inertia = kmeans.inertia_

    logger.info(f'Inertia: {inertia}')

    return ClusterResult(
        labels=cluster_labels,
        metadata={
            'method': 'kmeans',
            'n_clusters': n_clusters,
            'max_iter': max_iter,
            'inertia': inertia,
        },
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

    # run KMeans
    kmeans_params = {
        'n_clusters': args.n_clusters,
        'max_iter': args.max_iter,
    }
    kmeans_result = run_kmeans(X, **kmeans_params)
    df_clusters = pd.DataFrame(
        {
            'filename': df_embeddings['filename'],
            'cluster': kmeans_result.labels,
        }
    )

    # save results
    logger.info('Saving results')
    save_clustering_results(
        df_clusters,
        model_type='KMeans',
        out_dir=Path(args.outdir),
        embeddings_file=args.embeddings_file,
        cluster_metadata=kmeans_result.metadata,
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

    # kmeans args
    parser.add_argument(
        '--n_clusters',
        type=int,
        required=True,
        help='KMeans n_clusters parameter (required)',
    )
    parser.add_argument(
        '--max_iter',
        type=int,
        default=300,
        help='KMeans max_iter parameter (default: 300)',
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
