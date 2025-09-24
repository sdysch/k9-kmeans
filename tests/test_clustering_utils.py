# python
from pathlib import Path
import json
import pandas as pd
import pytest

# local
from k9kmeans.clustering.utils import (
    get_clusters_and_embeddings,
    save_clustering_results,
)


def test_save_clustering_results_creates_files(tmp_path: Path):
    cluster_df = pd.DataFrame({'filename': ['a.jpg', 'b.jpg'], 'cluster': [0, 1]})
    out_dir = tmp_path / "experiment1"
    model_type = "KMeans"
    embeddings_file = tmp_path / "embeddings.parquet"
    embeddings_file.write_text("dummy")

    metadata = {'n_clusters': 2}

    save_clustering_results(
        cluster_df, model_type, str(embeddings_file), out_dir, metadata
    )

    # Assertions
    result_file = out_dir / "cluster_results.csv"
    meta_file = out_dir / "params.json"

    assert result_file.exists()
    assert meta_file.exists()

    # Check CSV contents
    df_loaded = pd.read_csv(result_file)
    assert df_loaded.equals(cluster_df)

    # Check JSON contents
    with open(meta_file, 'r') as f:
        meta_loaded = json.load(f)
    assert meta_loaded['model_type'] == model_type
    assert meta_loaded['cluster_metadata'] == metadata
    assert str(Path(embeddings_file).resolve()) == meta_loaded['embeddings_file']


def test_get_clusters_and_embeddings(tmp_path: Path, monkeypatch):
    # Setup fake embeddings loader
    def fake_load_embeddings(path):
        return pd.DataFrame(
            {'filename': ['a.jpg', 'b.jpg'], 'embedding': [[0.1, 0.2], [0.3, 0.4]]}
        )

    monkeypatch.setattr(
        'k9kmeans.clustering.utils.load_embeddings', fake_load_embeddings
    )

    # Create experiment directory and files
    exp_dir = tmp_path / "exp1"
    exp_dir.mkdir()
    cluster_df = pd.DataFrame({'filename': ['a.jpg', 'b.jpg'], 'cluster': [0, 1]})
    embeddings_file = tmp_path / "emb.parquet"
    embeddings_file.write_text("dummy")  # just a path

    save_clustering_results(
        cluster_df, "KMeans", str(embeddings_file), exp_dir, {'n_clusters': 2}
    )

    # Run
    df_clusters, df_embeddings = get_clusters_and_embeddings(str(exp_dir))

    # Assertions
    assert isinstance(df_clusters, pd.DataFrame)
    assert isinstance(df_embeddings, pd.DataFrame)
    assert df_clusters.shape[0] == df_embeddings.shape[0]


def test_get_clusters_and_embeddings_errors(tmp_path: Path):
    # Non-existent directory
    with pytest.raises(ValueError):
        get_clusters_and_embeddings(str(tmp_path / "nonexistent"))

    # Missing CSV
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "params.json").write_text('{}')
    with pytest.raises(ValueError):
        get_clusters_and_embeddings(str(exp_dir))

    # Missing JSON
    (exp_dir / "cluster_results.csv").write_text("filename,cluster\n")
    (exp_dir / "params.json").unlink(missing_ok=True)
    with pytest.raises(ValueError):
        get_clusters_and_embeddings(str(exp_dir))
