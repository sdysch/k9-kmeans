# K9 Kmeans
[![Install](https://github.com/sdysch/k9-kmeans/actions/workflows/install.yml/badge.svg)](https://github.com/sdysch/k9-kmeans/actions/workflows/install.yml)

Fun (hopefully) side project clustering the far too many photos that I have of my dog
## Install
```bash
conda create -n k9kmeans python=3.12
conda activate k9kmeans
pip install -e .
```

### Install optional dependencies
```bash
pip install -e '.[dev]'
```

## Visualisation
```
run_dashboard --csv data/results_initial.csv
```

where `data/results_initial.csv` is a CSV file with the following columns:
- `filepath`: full path to the image file
- `cluster`: cluster label assigned to this image

## TODO
- [ ] Cleanup:
	- [ ] Pictures of Max with other dogs:
		- Initially cleanup by manually selecting pictures of just Max
		- Stretch goal would be to use a pre-trained model to identify pictures of multiple dogs, and filter these out
- [ ] Clustering:
	- [ ] Experiment with different clustering algorithms:
		-[ ] kmeans
		-[ ] DBSCAN
		-[ ] spectral clustering?
- [ ] Dimensionality reduction:
	- [ ] Experiment with different techniques, not just PCA
		- [ ] t-SNE
		- [ ] UMAP
- [ ] Add CI tests for core functions

<img src="Max.jpg" alt="Max" style="width:50%;">
