# python
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# local
from k9kmeans.image_utils.processing import load_and_preprocess_image, get_embeddings
from k9kmeans.logging import setup_logger


logger = setup_logger(__name__)


def get_filenames(base_path: Path, limit: Optional[int] = None) -> list[Path]:
    """Return a list of image file paths under base_path, optionally limited."""
    if not base_path.is_dir():
        raise ValueError(f'Provided image_dir "{base_path}" is not a directory')

    exts = {'.jpg', '.jpeg', '.png', '.webp'}
    files = [p for p in base_path.iterdir() if p.suffix.lower() in exts]

    if limit:
        files = files[:limit]
        logger.info(f'Limiting to first {limit} images')

    if not files:
        raise ValueError(f'No images found in directory {base_path}')

    logger.info(f'Found {len(files)} images in directory {base_path}')
    return files


def load_model_and_processor(
    model_name: str, device: str
) -> Tuple[CLIPModel, CLIPProcessor]:
    """Load CLIP model and processor on the given device."""
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    logger.info(f'Loaded model {model_name} on device {device}')
    return model, processor


def process_batches(
    filenames: list[Path],
    batch_size: int,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
) -> Tuple[list[str], NDArray[np.float32]]:
    """Load images in batches, compute embeddings, and return results."""
    added_filenames: list[str] = []
    all_embeddings: list[NDArray[np.float32]] = []

    for i in tqdm(range(0, len(filenames), batch_size), desc='Processing batches'):
        batch_paths = filenames[i : i + batch_size]
        images = []

        for path in batch_paths:
            try:
                images.append(load_and_preprocess_image(str(path)))
                added_filenames.append(str(path))
            except Exception as e:
                logger.warning(f'Error loading image {path}: {type(e).__name__}: {e}')
                continue

        if not images:
            continue

        emb = get_embeddings(images, processor, model, device=device)
        all_embeddings.append(emb)

    all_embeddings_array: NDArray[np.float32] = np.vstack(all_embeddings)

    assert len(added_filenames) == all_embeddings_array.shape[0], (
        f'Number of filenames ({len(added_filenames)}) and embeddings '
        f'({all_embeddings_array.shape[0]}) do not match'
    )

    return added_filenames, all_embeddings_array


def save_embeddings(
    filenames: list[str], embeddings: NDArray[np.float32], outfile: Path
) -> None:
    """Save embeddings and filenames to parquet file."""
    df = pd.DataFrame({'filename': filenames, 'embedding': list(embeddings)})
    df.to_parquet(outfile, index=False)
    logger.info(f'Saved embeddings to {outfile}')


def main(args: argparse.Namespace) -> None:
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(log_level)

    base_path = Path(args.image_dir)
    filenames = get_filenames(base_path, limit=args.limit)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, processor = load_model_and_processor('openai/clip-vit-base-patch32', device)

    added_filenames, all_embeddings_array = process_batches(
        filenames, args.batch_size, model, processor, device
    )

    save_embeddings(added_filenames, all_embeddings_array, Path(args.outfile))


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
