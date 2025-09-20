from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List
import io
import numpy as np
import torch


def load_and_preprocess_image(path: str) -> Image.Image:
    """Load an image from disk and convert to RGB."""
    img = Image.open(path).convert('RGB')
    return img


def get_embeddings(
    images: List[Image.Image],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str = 'cpu',
) -> np.ndarray:
    """Extract CLIP embeddings for a list of PIL images."""
    inputs = processor(images=images, return_tensors='pt', padding=True)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs.to(device))
    return embeddings.cpu().numpy()
