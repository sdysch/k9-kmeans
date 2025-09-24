from pathlib import Path
import numpy as np
import pandas as pd

from k9kmeans.pipeline.embeddings import get_filenames, save_embeddings


def test_get_filenames(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"dummy")
    (tmp_path / "b.png").write_bytes(b"dummy")
    (tmp_path / "not_an_image.txt").write_text("hello")

    filenames = get_filenames(tmp_path)
    assert all(
        f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} for f in filenames
    )
    assert len(filenames) == 2


def test_save_embeddings(tmp_path: Path):
    filenames = ["a.jpg", "b.jpg"]
    embeddings: NDArray[np.float32] = np.zeros((2, 512), dtype=np.float32)

    outfile = tmp_path / "embeddings.parquet"
    save_embeddings(filenames, embeddings, outfile)

    df = pd.read_parquet(outfile)
    assert list(df["filename"]) == filenames
    np.testing.assert_array_equal(np.array(df["embedding"].tolist()), embeddings)
