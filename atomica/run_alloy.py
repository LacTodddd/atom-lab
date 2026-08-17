import argparse
from pathlib import Path
from atomica.alloy import evaluate, brute_force_min, N_SITES, N_AU
from atomica.alloy_search import random_search, genetic_search, active_learning_search
from atomica.benchmark import run_alloy_benchmark
from atomica.plot import make_figures, write_metrics, KNOWN_MINIMA

METHODS = {"random": random_search, "genetic": genetic_search, "active": active_learning_search}

def main(argv=None):
    p = argparse.ArgumentParser(description="ATOMICA P2 Cu-Au alloy-ordering benchmark")
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)

    gt_path = Path(a.out) / "alloy_ground_truth.json"
    min_e, _, _ = brute_force_min(evaluate, N_SITES, N_AU, cache_path=gt_path)

    methods = {name: METHODS[name] for name in a.methods}
    run_alloy_benchmark(methods, list(range(a.seeds)), a.budget,
                        evaluate, N_SITES, N_AU, out_dir=a.out)
    known = {**KNOWN_MINIMA, N_SITES: min_e}
    make_figures(results_dir=a.out, out_dir=a.out, known_minima=known,
                 title_fmt="Cu-Au {n}-site ordering", xlabel="MACE evaluations")
    write_metrics(results_dir=a.out, out_dir=a.out, known_minima=known)

if __name__ == "__main__":
    main()
