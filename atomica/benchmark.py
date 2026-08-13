import json
from pathlib import Path
import numpy as np
from atomica.potential import relax
from atomica.search import random_search, genetic_search, active_learning_search

METHODS = {
    "random": random_search,
    "genetic": genetic_search,
    "active": active_learning_search,
}

def run_benchmark(n_values, seeds, budget, methods=METHODS, out_dir="results"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for n in n_values:
        for name, fn in methods.items():
            for seed in seeds:
                history, best = fn(n, budget, seed, relax)
                path = out / f"{name}_N{n}_seed{seed}.json"
                path.write_text(json.dumps({
                    "method": name, "n": int(n), "seed": int(seed), "budget": int(budget),
                    "history": [[int(s), float(e)] for s, e in history],
                    "best_energy": float(history[-1][1]),
                    "best_positions": np.asarray(best).tolist(),
                }))
                written.append(str(path))
    return written
