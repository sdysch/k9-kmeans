# python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from argparse import ArgumentParser

# local
from k9kmeans.clustering.preprocessing import preprocess_embeddings, load_embeddings


def dbscan_k_distance_elbow(embeddings, min_samples_list):

    records = []
    for k in min_samples_list:
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(embeddings)
        distances, _ = neigh.kneighbors(embeddings)
        k_distances = np.sort(distances[:, -1])
        for i, d in enumerate(k_distances):
            records.append({'Point': i, 'k_distance': d, 'min_samples': k})

    return pd.DataFrame(records)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--embeddings_file', type=str, required=True)
    args = parser.parse_args()

    df_embeddings = load_embeddings(args.embeddings_file)
    X = np.stack(df_embeddings['embedding'].to_numpy())
    X = preprocess_embeddings(X)

    min_samples_list = [3, 5, 10, 15, 20]
    df = dbscan_k_distance_elbow(X, min_samples_list)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x='Point', y='k_distance', hue='min_samples', palette='tab10')
    fig.savefig('dbscan_k_distance_elbow.png')
