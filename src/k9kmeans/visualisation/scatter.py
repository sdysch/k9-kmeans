# python
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# local
from k9kmeans.logging import setup_logger
from k9kmeans.clustering.utils import get_clusters_and_embeddings
from k9kmeans.visualisation.dimensionality_reduction import pca, tsne


logger = setup_logger(__name__)


def make_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    palette: str,
    save_path: Path,
    fig_sfx: str,
) -> None:
    """Make and save a scatter plot"""
    logging.info('Making scatter plot...')

    hue_order = sorted(df[hue_col].unique(), key=lambda x: (x == '-1', x))

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        hue_order=hue_order,
        palette=palette,
        s=60,
        alpha=0.8,
        ax=ax,
    )

    fullpath = Path(save_path) / fig_sfx
    fig.savefig(fullpath, dpi=300)
    logger.info(f'Saved scatter plot to {fullpath}')


def process_experiment(exp_path: str, args: argparse.Namespace) -> None:
    """Run dimensionality reduction and scatter plot for one experiment"""
    clusters, embeddings = get_clusters_and_embeddings(exp_path)
    clusters = clusters['cluster'].to_numpy()
    embeddings = np.stack(embeddings['embedding'].to_numpy())

    if args.method == 'pca':
        reduced_embeddings = pca(embeddings, n_components=2)
    elif args.method == 'tsne':
        reduced_embeddings = tsne(
            embeddings, n_components=2, perplexity=args.perplexity
        )
    else:
        raise ValueError(f'Unknown dimensionality reduction method: {args.method}')

    logging.info(f'Reduced embeddings shape: {reduced_embeddings.shape}')

    # make plot
    df_comb = pd.DataFrame(
        {
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'cluster': clusters,
        }
    )

    if not args.include_noise:
        df_comb = df_comb[df_comb['cluster'] != -1]
    else:
        logging.info('Including noise points in the scatter plot')

    df_comb['cluster'] = df_comb['cluster'].astype(str)

    fig_sfx = (
        'scatter_plot'
        + ('_with_noise' if args.include_noise else '')
        + ('_' + args.method if args.method else '')
        + '.png'
    )

    make_scatter(
        df_comb,
        x_col='x',
        y_col='y',
        hue_col='cluster',
        palette=args.palette,
        save_path=Path(exp_path),
        fig_sfx=fig_sfx,
    )


def main(args: argparse.Namespace) -> None:

    logger.info(f'Making scatter plots for {len(args.experiments)} experiments...')

    for exp in args.experiments:
        process_experiment(exp, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--experiments',
        type=str,
        nargs='+',
        required=True,
        help='Directory that cluster experiments are stored in',
    )
    parser.add_argument(
        '--include-noise',
        action='store_true',
        default=False,
        help='Include noise points in the scatter plot, relevant for DBSCAN',
    )
    parser.add_argument(
        '--method',
        type=str,
        default='pca',
        choices=['pca', 'tsne'],
        help='Dimensionality reduction method to use for scatter plot default pca',
    )
    parser.add_argument(
        '--palette',
        type=str,
        default='tab20',
        help='Color palette to use for scatter plot, default tab20',
    )

    # tsne options
    parser.add_argument(
        '--perplexity',
        type=float,
        default=30.0,
        help='Perplexity parameter for t-SNE, default 30.0. Only relevant if method is tsne',
    )

    args = parser.parse_args()
    main(args)
