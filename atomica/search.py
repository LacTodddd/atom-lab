import numpy as np


def random_cluster(n: int, rng, min_sep: float = 0.8) -> np.ndarray:
    R = 0.5 * n ** (1 / 3) + 1.0
    pts = []
    while len(pts) < n:
        p = rng.uniform(-R, R, size=3)
        if np.linalg.norm(p) > R:
            continue
        if any(np.linalg.norm(p - q) < min_sep for q in pts):
            continue
        pts.append(p)
    return np.array(pts)


def random_search(n, budget, seed, relax):
    rng = np.random.default_rng(seed)
    best_e, best_x = np.inf, None
    history = []
    for i in range(budget):
        x, e = relax(random_cluster(n, rng))
        if e < best_e:
            best_e, best_x = e, x
        history.append((i + 1, best_e))
    return history, best_x
