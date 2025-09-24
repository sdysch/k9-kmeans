#!/bin/bash

for k in {2..20}; do
  python -m k9kmeans.clustering.kmeans \
    --outdir "data/kmeans/kmeans_experiment_n_clusters_${k}" \
    --embeddings_file data/embeddings_all.parquet \
    --n_clusters ${k}
done
