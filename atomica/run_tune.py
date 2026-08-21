import argparse, json
from pathlib import Path
from atomica.strategist import tune, compare, random_proposer, llm_proposer, MODEL
from atomica.llm import make_llm_arm, run_llm_arm
import numpy as np

def make_llm_proposer(model):
    """Return an llm_proposer bound to a real client, or None if no credential/SDK."""
    return make_llm_arm(llm_proposer, model, "run_tune")

def _best_over_trajectories(make_proposer, trajectories, rounds, tune_seeds, budget):
    best_overall, best_score = None, None
    for t in range(trajectories):
        proposer = make_proposer(t)
        params, trace = tune(proposer, rounds, tune_seeds, budget, seed=t)
        s = min(trace, key=lambda r: (r["mean_best"], r["mean_evals"]))
        if best_score is None or (s["mean_best"], s["mean_evals"]) < best_score:
            best_score, best_overall = (s["mean_best"], s["mean_evals"]), params
    return best_overall

def main(argv=None):
    p = argparse.ArgumentParser(description="ATOMICA P3 LLM-vs-random hyperparameter tuning (LJ-38)")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--tune-seeds", type=int, default=2)
    p.add_argument("--eval-seeds", type=int, default=5)
    p.add_argument("--budget", type=int, default=120)
    p.add_argument("--trajectories", type=int, default=3)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)
    tune_seeds = list(range(a.tune_seeds))
    eval_seeds = list(range(100, 100 + a.eval_seeds))     # disjoint from tune seeds

    best = {}
    best["random"] = _best_over_trajectories(
        lambda t: random_proposer(np.random.default_rng(1000 + t)),
        a.trajectories, a.rounds, tune_seeds, a.budget)

    llm = make_llm_proposer(a.model)
    if llm is not None:
        res = run_llm_arm("run_tune", lambda: _best_over_trajectories(
            lambda t: llm, a.trajectories, a.rounds, tune_seeds, a.budget))
        if res is not None:
            best["llm"] = res

    comparison = compare(best, eval_seeds, a.budget)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "tune_report.json").write_text(json.dumps(
        {"best_params": best, "comparison": comparison,
         "config": {"rounds": a.rounds, "tune_seeds": tune_seeds, "eval_seeds": eval_seeds,
                    "budget": a.budget, "trajectories": a.trajectories, "model": a.model}},
        indent=2))
    print(json.dumps({k: {"mean_best": v["mean_best"], "mean_evals": v["mean_evals"]}
                      for k, v in comparison.items()}, indent=2))

if __name__ == "__main__":
    main()
