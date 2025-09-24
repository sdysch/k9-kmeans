from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Any


@dataclass
class ClusterResult:
    labels: NDArray[np.int32]
    metadata: dict[str, Any] | None = None
    silhouette_score: float | None = None
