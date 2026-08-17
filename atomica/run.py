import argparse
from atomica.benchmark import run_benchmark, METHODS
from atomica.plot import make_figures, write_metrics

def main(argv=None):
    p = argparse.ArgumentParser(description="ATOMICA LJ-cluster search benchmark")
    p.add_argument("--n", type=int, nargs="+", default=[13, 38])
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--seeds", type=int, default=5, help="number of seeds (0..seeds-1)")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)
    methods = {name: METHODS[name] for name in a.methods}
    run_benchmark(a.n, list(range(a.seeds)), a.budget, methods=methods, out_dir=a.out)
    make_figures(results_dir=a.out, out_dir=a.out)
    write_metrics(a.out, a.out)

if __name__ == "__main__":
    main()
