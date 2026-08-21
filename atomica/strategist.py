import numpy as np
from atomica.potential import relax
from atomica.search import active_learning_search
from atomica.plot import KNOWN_MINIMA, evals_to_target
from atomica.llm import call_strict_tool

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

def random_proposer(rng):
    def propose(history):
        return {"k_acq": float(rng.uniform(*PARAM_SPACE["k_acq"])),
                "pool": int(rng.choice(PARAM_SPACE["pool"])),
                "n_init": int(rng.choice(PARAM_SPACE["n_init"]))}
    return propose

def tune(proposer, rounds, tune_seeds, budget, seed=0, n=38):
    rng = np.random.default_rng(seed)
    fallback_draw = random_proposer(rng)
    trace = []
    for r in range(rounds):
        fallback = False
        try:
            params = validate_params(proposer(trace))
        except ValueError:
            params = validate_params(fallback_draw(trace))
            fallback = True
        score = score_params(params, tune_seeds, budget, n)
        trace.append({**score, "round": r, "fallback": fallback})
    best = min(trace, key=lambda t: (t["mean_best"], t["mean_evals"]))["params"]
    return best, trace

MODEL = "claude-sonnet-5"

_TOOL = {
    "name": "propose_params",
    "description": "Propose the next hyperparameters to try for the active-learning search.",
    "input_schema": {
        "type": "object",
        "properties": {
            "k_acq": {"type": "number", "description": "LCB weight in [0,3]"},
            "pool": {"type": "integer", "description": "candidate pool size: 40, 80, or 160"},
            "n_init": {"type": "integer", "description": "initial samples: 5, 10, or 20"},
        },
        "required": ["k_acq", "pool", "n_init"],
        "additionalProperties": False,
    },
    "strict": True,
}

def build_prompt(history):
    lines = ["You tune an active-learning search that minimizes cluster energy in as few",
             "evaluations as possible. Parameter space: k_acq in [0,3] (float),",
             "pool in {40,80,160}, n_init in {5,10,20}. Lower mean_best is better;",
             "break ties by lower mean_evals. History of what has been tried:"]
    if not history:
        lines.append("(none yet — propose a sensible first configuration)")
    for t in history:
        p = t["params"]
        lines.append(f"- k_acq={p['k_acq']:.2f} pool={p['pool']} n_init={p['n_init']}"
                     f" -> mean_best={t['mean_best']:.3f}, mean_evals={t['mean_evals']:.1f}")
    lines.append("Call propose_params with the next configuration to try.")
    return "\n".join(lines)

def llm_proposer(client, model=MODEL):
    def propose(history):
        return call_strict_tool(client, model, _TOOL, build_prompt(history))
    return propose

def compare(best_by_tuner, eval_seeds, budget, n=38):
    everyone = {**best_by_tuner, "default": DEFAULT_PARAMS}
    return {name: score_params(params, eval_seeds, budget, n) for name, params in everyone.items()}
