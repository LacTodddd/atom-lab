"""ATOMICA P5 CLI: does a paper-reading LLM predict the unexplored optimum better than a
paper-blind baseline? (Cu-Au research-gap extrapolation)."""
import argparse, json
from pathlib import Path
from atomica.litreview_world import build_world, generate_papers
from atomica.litreview import (
    baseline_reviewer, heuristic_reviewer, llm_reviewer, review_batch, score, MODEL,
)
from atomica.llm import make_llm_arm, run_llm_arm

def make_llm_reviewer(model):
    """Return an llm_reviewer bound to a real client, or None if no credential/SDK."""
    return make_llm_arm(llm_reviewer, model, "run_litreview")

def main(argv=None):
    p = argparse.ArgumentParser(description="ATOMICA P5 literature-gap extrapolation benchmark (Cu-Au)")
    p.add_argument("--n-papers", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)

    world = build_world(cache_path=str(Path(a.out) / "alloy_world.json"))
    papers = generate_papers(world, a.n_papers, seed0=a.seed)

    arms = {}
    arms["baseline"] = score(papers, review_batch(papers, baseline_reviewer))
    arms["heuristic"] = score(papers, review_batch(papers, heuristic_reviewer))

    llm = make_llm_reviewer(a.model)
    if llm is not None:
        res = run_llm_arm("run_litreview", lambda: score(papers, review_batch(papers, llm)))
        if res is not None:
            arms["llm"] = res

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "litreview_report.json").write_text(json.dumps(
        {"arms": arms, "config": {"n_papers": a.n_papers, "seed": a.seed, "model": a.model}}, indent=2))
    print(json.dumps({k: {"accuracy": round(v["accuracy"], 3), "recall": round(v["recall"], 3)}
                      for k, v in arms.items()}, indent=2))

if __name__ == "__main__":
    main()
