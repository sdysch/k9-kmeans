# python
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import logging
import numpy as np
import os
import pandas as pd
import sys
import torch

# local
from k9kmeans.utils import load_and_preprocess_image, get_embeddings
from k9kmeans.logging import setup_logger


# TODO model choice?
MODEL_NAME = 'openai/clip-vit-base-patch32'


def main(args: argparse.Namespace) -> None:

    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(__name__, level=log_level)
    logging.basicConfig(
        level=log_level,
    )

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

    # image preprocessing
    images = [load_and_preprocess_image(f) for f in filenames]

    # load model and processor
    model_name = MODEL_NAME  # later from args?
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # get embeddings
    all_embeddings = []

    for i in tqdm(range(0, len(images), args.batch_size), desc='Extracting embeddings'):
        batch = images[i : i + args.batch_size]
        emb = get_embeddings(batch, processor, model, device=device)
        all_embeddings.append(emb)

    all_embeddings = np.vstack(all_embeddings)

    # save to csv
    df = pd.DataFrame({'filename': filenames, 'embedding': list(all_embeddings)})

    df.to_parquet(args.outfile, index=False)


if __name__ == '__main__':

    import argparse

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
