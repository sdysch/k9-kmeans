# python
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, cast
import io
import numpy as np
from numpy.typing import NDArray
import torch

# local
from k9kmeans.logging import setup_logger

logger = setup_logger(__name__)


def load_and_preprocess_image(path: str) -> Image.Image:
    """Load an image from disk and convert to RGB."""
    img = Image.open(path).convert('RGB')
    logger.debug(f'Loaded image {path} with size {img.size} and mode {img.mode}')
    return img


def get_embeddings(
    images: List[Image.Image],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str = 'cpu',
) -> NDArray[np.float32]:
    """Extract CLIP embeddings for a list of PIL images."""
    inputs = processor(images=images, return_tensors='pt', padding=True)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs.to(device))
    return cast(
        NDArray[np.float32], embeddings.cpu().numpy().astype(np.float32, copy=False)
    )
