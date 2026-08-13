import numpy as np
from atomica.potential import relax
from atomica.search import random_cluster, random_search


def test_random_cluster_shape_and_separation():
    rng = np.random.default_rng(0)
    x = random_cluster(6, rng, min_sep=0.8)
    assert x.shape == (6, 3)
    from scipy.spatial.distance import pdist
    assert pdist(x).min() > 0.8


def test_random_search_history_valid():
    hist, best = random_search(2, budget=15, seed=0, relax=relax)
    assert len(hist) == 15
    steps = [h[0] for h in hist]
    energies = [h[1] for h in hist]
    assert steps == list(range(1, 16))                     # 1..budget
    assert all(energies[i] >= energies[i + 1] - 1e-9 for i in range(len(energies) - 1))  # non-increasing
    assert energies[-1] < -0.9                             # N=2 global min is -1.0
    assert best.shape == (2, 3)
