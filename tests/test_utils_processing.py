import pytest
from PIL import Image
from k9kmeans.image_utils.processing import load_and_preprocess_image


def test_load_and_preprocess_image(tmp_path):
    # create a small dummy image
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (32, 32)).save(img_path)

    img = load_and_preprocess_image(str(img_path))
    assert isinstance(img, Image.Image)
    assert img.size == (32, 32)
    assert img.mode == "RGB"
