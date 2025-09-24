import itertools
import subprocess
import os
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed

# Parameter ranges
eps_values = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
metrics = ['cosine', 'euclidean']
min_samples_values = [3, 5, 10, 15, 20]

embeddings_file = 'data/embeddings_all.parquet'


# Function to determine safe number of workers based on system load
def get_max_workers(load_factor=0.7):
    cpu_count = os.cpu_count() or 4
    available_cpus = max(1, int(cpu_count * load_factor))
    return available_cpus


def run_dbscan(eps, metric, min_samples):
    outdir = f'data/dbscan/dbscan_eps_{eps}_min_sample_{min_samples}_metric_{metric}'
    cmd = [
        'python',
        '-m',
        'k9kmeans.clustering.dbscan',
        '--outdir',
        outdir,
        '--embeddings_file',
        embeddings_file,
        '--eps',
        str(eps),
        '--min_samples',
        str(min_samples),
        '--metric',
        metric,
    ]
    print(f'Running: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)
    return outdir


# Create all combinations
combinations = list(itertools.product(eps_values, metrics, min_samples_values))

# Determine max workers based on CPU and load
max_workers = get_max_workers(load_factor=0.7)
print(f'Running up to {max_workers} parallel jobs.')

# Run in parallel using processes
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = [
        executor.submit(run_dbscan, eps, metric, min_samples)
        for eps, metric, min_samples in combinations
    ]

    for future in as_completed(futures):
        try:
            outdir = future.result()
            print(f'Finished: {outdir}')
        except subprocess.CalledProcessError as e:
            print(f'Error running job: {e}')
