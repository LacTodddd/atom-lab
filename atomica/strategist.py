import numpy as np
from atomica.potential import relax
from atomica.search import active_learning_search
from atomica.plot import KNOWN_MINIMA, evals_to_target

PARAM_SPACE = {"k_acq": (0.0, 3.0), "pool": [40, 80, 160], "n_init": [5, 10, 20]}
DEFAULT_PARAMS = {"k_acq": 1.0, "pool": 100, "n_init": 10}

def validate_params(raw):
    if not isinstance(raw, dict):
        raise ValueError(f"params not a dict: {raw!r}")
    try:
        k = float(raw["k_acq"]); pool = int(raw["pool"]); n = int(raw["n_init"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"unparseable params: {raw!r}") from e
    lo, hi = PARAM_SPACE["k_acq"]
    k = min(max(k, lo), hi)
    pool = min(PARAM_SPACE["pool"], key=lambda x: abs(x - pool))
    n = min(PARAM_SPACE["n_init"], key=lambda x: abs(x - n))
    return {"k_acq": k, "pool": pool, "n_init": n}

def score_params(params, tune_seeds, budget, n=38):
    target = KNOWN_MINIMA[n]
    bests, evals = [], []
    for seed in tune_seeds:
        history, _ = active_learning_search(n, budget, seed, relax, **params)
        bests.append(history[-1][1])
        e = evals_to_target(history, target)
        evals.append(e if e is not None else budget)
    return {"mean_best": float(np.mean(bests)),
            "mean_evals": float(np.mean(evals)),
            "params": params}
