# python
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# local
from k9kmeans.logging import setup_logger
from k9kmeans.clustering.utils import get_clusters_and_embeddings


logger = setup_logger(__name__)
