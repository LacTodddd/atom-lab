import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Confirm exact values against the Cambridge Cluster Database.
KNOWN_MINIMA = {13: -44.326801, 38: -173.928427}

def success_rate(histories, target, tol=0.01):
    hits = sum(1 for h in histories if h[-1][1] <= target + tol)
    return hits / len(histories)

def evals_to_target(history, target, tol=0.01):
    for step, energy in history:
        if energy <= target + tol:
            return step
    return None

def _load(results_dir):
    runs = defaultdict(list)  # (n, method) -> list of run dicts
    for p in Path(results_dir).glob("*.json"):
        d = json.loads(p.read_text())
        runs[(d["n"], d["method"])].append(d)
    return runs

def make_figures(results_dir="results", out_dir="results"):
    runs = _load(results_dir)
    ns = sorted({n for (n, _) in runs})
    written = []
    for n in ns:
        plt.figure()
        for (rn, method), group in sorted(runs.items()):
            if rn != n:
                continue
            budget = group[0]["budget"]
            curves = np.array([[e for _, e in d["history"]] for d in group])  # (seeds, budget)
            mean, std = curves.mean(0), curves.std(0)
            x = np.arange(1, budget + 1)
            plt.plot(x, mean, label=method)
            plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        if n in KNOWN_MINIMA:
            plt.axhline(KNOWN_MINIMA[n], ls="--", color="k", label="global min")
        plt.xlabel("relaxations"); plt.ylabel("best energy"); plt.title(f"LJ-{n}"); plt.legend()
        path = str(Path(out_dir) / f"convergence_N{n}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
        written.append(path)
    return written
