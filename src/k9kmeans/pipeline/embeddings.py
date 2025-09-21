# python
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import logging
import numpy as np
from numpy.typing import NDArray
import os
import pandas as pd
import sys
import torch
import argparse

# local
from k9kmeans.image_utils.processing import load_and_preprocess_image, get_embeddings
from k9kmeans.logging import setup_logger


# TODO model choice?
MODEL_NAME = 'openai/clip-vit-base-patch32'


def main(args: argparse.Namespace) -> None:

    # setup logger
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(__name__, level=log_level)

    if args.debug:
        logger.debug('Debug logging enabled')
    logger.info(f'Using image_dir: {args.image_dir}')

    # extract all image filenames
    base_path = args.image_dir
    if not os.path.isdir(base_path):
        logger.error(f'Provided image_dir "{base_path}" is not a directory')
        sys.exit(1)

    filenames = [
        os.path.join(base_path, f)
        for f in os.listdir(base_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ]

    if args.limit:
        filenames = filenames[: args.limit]
        logger.info(f'Limiting to first {args.limit} images')

    if len(filenames) == 0:
        logger.error(f'No images found in directory {base_path}')
        sys.exit(1)

    logger.info(f'Found {len(filenames)} images in directory {base_path}')

    # load model and processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = MODEL_NAME  # could make this an arg later
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    added_filenames: list[str] = []
    all_embeddings: list[NDArray[np.float32]] = []

    # iterate in batches
    for i in tqdm(
        range(0, len(filenames), args.batch_size),
        desc='Loading images and computing embeddings',
    ):
        batch_paths = filenames[i : i + args.batch_size]
        images = []

        for path in batch_paths:
            try:
                images.append(load_and_preprocess_image(path))
                added_filenames.append(path)
            except Exception as e:
                logger.warning(f'Error loading image {path}: {e}')
                continue

        # skip empty batches
        if not images:
            continue

        # get embeddings for this batch
        emb = get_embeddings(images, processor, model, device=device)
        all_embeddings.append(emb)

    # combine all embeddings
    all_embeddings_array: NDArray[np.float32] = np.vstack(all_embeddings)

    assert len(added_filenames) == all_embeddings_array.shape[0], (
        f'Number of filenames ({len(added_filenames)}) and embeddings '
        f'({all_embeddings_array.shape[0]}) do not match'
    )

    # save results
    df = pd.DataFrame(
        {'filename': added_filenames, 'embedding': list(all_embeddings_array)}
    )

    df.to_parquet(args.outfile, index=False)
    logger.info(f'Saved embeddings to {args.outfile}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--outfile',
        type=str,
        required=True,
        help='Output parquet file that embeddings are stored in',
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='Path to directory containing images',
    )
    parser.add_argument(
        '--batch_size', type=int, default=32, help='Number of batches to process'
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument(
        '--limit', type=int, help='Limit number of images to process (for testing)'
    )

    args = parser.parse_args()
    main(args)
