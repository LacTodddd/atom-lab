import numpy as np
from scipy.spatial.distance import pdist

def distance_histogram(positions: np.ndarray, bins: int = 30, r_max: float = 8.0) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 2:
        raise ValueError("distance_histogram needs at least 2 atoms")
    d = np.clip(pdist(positions), 0.0, r_max)  # clip to ensure all distances in [0, r_max]
    hist, _ = np.histogram(d, bins=bins, range=(0.0, r_max))
    return hist / hist.sum()
