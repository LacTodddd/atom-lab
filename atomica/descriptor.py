import numpy as np
from scipy.spatial.distance import pdist

def distance_histogram(positions: np.ndarray, bins: int = 30, r_max: float = 8.0) -> np.ndarray:
    d = pdist(np.asarray(positions, dtype=float))          # all pairwise distances
    hist, _ = np.histogram(d, bins=bins, range=(0.0, r_max))
    total = hist.sum()
    if total == 0:
        return np.zeros(bins)
    return hist / total
