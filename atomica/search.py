import numpy as np
from sklearn.ensemble import RandomForestRegressor
from atomica.descriptor import distance_histogram


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


def cut_and_splice(a, b, rng):
    n = len(a)
    a = a - a.mean(0)
    b = b - b.mean(0)
    normal = rng.normal(size=3)
    normal /= np.linalg.norm(normal)
    a_sorted = a[np.argsort(a @ normal)]
    b_sorted = b[np.argsort(b @ normal)]
    k = int(rng.integers(1, n))
    return np.vstack([a_sorted[:k], b_sorted[k:]])


def mutate(x, rng, sigma=0.3, frac=0.3):
    x = x.copy()
    m = rng.random(len(x)) < frac
    if m.any():
        x[m] += rng.normal(scale=sigma, size=(int(m.sum()), 3))
    return x


def genetic_search(n, budget, seed, relax, pop_size=10):
    rng = np.random.default_rng(seed)
    used = 0
    history = []
    pop = []  # list of (energy, positions)
    best_e, best_x = np.inf, None

    def record(e, x):
        nonlocal best_e, best_x, used
        used += 1
        if e < best_e:
            best_e, best_x = e, x
        history.append((used, best_e))

    for _ in range(min(pop_size, budget)):
        x, e = relax(random_cluster(n, rng))
        pop.append((e, x))
        record(e, x)

    while used < budget:
        pop.sort(key=lambda t: t[0])
        parents = pop[: max(2, pop_size // 2)]
        (_, pa), (_, pb) = (parents[int(rng.integers(len(parents)))] for _ in range(2))
        child = mutate(cut_and_splice(pa, pb, rng), rng)
        x, e = relax(child)
        record(e, x)
        pop.append((e, x))
        pop.sort(key=lambda t: t[0])
        pop = pop[:pop_size]                                # keep the fittest

    return history, best_x


def active_learning_search(n, budget, seed, relax, n_init=10, pool=100, k_acq=1.0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    best_e, best_x = np.inf, None
    history = []
    used = 0

    def record(x, e):
        nonlocal best_e, best_x, used
        used += 1
        X.append(distance_histogram(x))
        y.append(e)
        if e < best_e:
            best_e, best_x = e, x
        history.append((used, best_e))

    for _ in range(min(n_init, budget)):
        x, e = relax(random_cluster(n, rng))
        record(x, e)

    while used < budget:
        model = RandomForestRegressor(n_estimators=100, random_state=seed)
        model.fit(np.array(X), np.array(y))
        cands = [random_cluster(n, rng) for _ in range(pool // 2)]
        cands += [mutate(best_x, rng) for _ in range(pool - pool // 2)]
        D = np.array([distance_histogram(c) for c in cands])
        preds = np.stack([est.predict(D) for est in model.estimators_])  # (n_trees, pool)
        acq = preds.mean(0) - k_acq * preds.std(0)         # lower-confidence-bound (minimizing energy)
        x, e = relax(cands[int(np.argmin(acq))])
        record(x, e)

    return history, best_x
