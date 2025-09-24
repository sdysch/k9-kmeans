# python
import pandas as pd
import pytest
import ast

# local
from k9kmeans.clustering.preprocessing import load_embeddings


def test_load_embeddings_csv(tmp_path):
    df = pd.DataFrame({'embedding': [[0.1, 0.2], [0.3, 0.4]]})
    file = tmp_path / "emb.csv"
    df.to_csv(file, index=False)

    df_loaded = load_embeddings(str(file))
    df_loaded['embedding'] = df_loaded['embedding'].apply(ast.literal_eval)
    pd.testing.assert_frame_equal(df, df_loaded)


def test_load_embeddings_parquet(tmp_path):
    df = pd.DataFrame({'embedding': [[0.1, 0.2], [0.3, 0.4]]})
    file = tmp_path / "emb.parquet"
    df.to_parquet(file, index=False)

    df_loaded = load_embeddings(str(file))
    pd.testing.assert_frame_equal(df, df_loaded)


def test_load_embeddings_unsupported_format(tmp_path):
    file = tmp_path / "emb.txt"
    file.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_embeddings(str(file))


def test_load_embeddings_missing_column(tmp_path):
    df = pd.DataFrame({'foo': [1, 2]})
    file = tmp_path / "emb.csv"
    df.to_csv(file, index=False)
    with pytest.raises(
        ValueError, match='DataFrame must contain an "embedding" column'
    ):
        load_embeddings(str(file))


def test_load_embeddings_unsupported_format(tmp_path):
    file = tmp_path / "emb.txt"
    file.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_embeddings(str(file))


def test_load_embeddings_missing_column(tmp_path):
    df = pd.DataFrame({'foo': [1, 2]})
    file = tmp_path / "emb.csv"
    df.to_csv(file, index=False)
    with pytest.raises(
        ValueError, match='DataFrame must contain an "embedding" column'
    ):
        load_embeddings(str(file))
