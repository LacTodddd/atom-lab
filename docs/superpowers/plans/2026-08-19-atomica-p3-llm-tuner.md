# ATOMICA P3 — LLM Strategist (Hyperparameter Tuner) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an LLM tune `active_learning_search`'s hyperparameters on LJ-38 and measure — cheat-proof — whether it beats random tuning and the Slice-1 default.

**Architecture:** A new `atomica/strategist.py` reuses Slice 1's `active_learning_search` (unchanged; it already exposes `n_init`/`pool`/`k_acq`). A shared R-round `tune` loop takes a *proposer* (random or LLM) that returns three bounded parameters as validated JSON; every proposal is scored by running the real search on LJ-38. A CLI runs both tuners plus the default and reports the comparison. The LLM only proposes parameters — never code, never physics.

**Tech Stack:** Python 3.13, `anthropic` SDK (structured output), reusing `numpy`, `ase`, `scikit-learn` via the existing `atomica` package.

**Spec:** `docs/superpowers/specs/2026-08-19-atomica-p3-llm-tuner-design.md`

## Global Constraints

- **Reuse `atomica/search.py::active_learning_search(n, budget, seed, relax, n_init=10, pool=100, k_acq=1.0)` unchanged.** Do not modify Slice 1 / P2 files.
- **Parameter space (bounded, validated):** `k_acq ∈ [0.0, 3.0]` (float), `pool ∈ {40, 80, 160}` (int), `n_init ∈ {5, 10, 20}` (int).
- **Default reference** = exact Slice-1 params `{k_acq: 1.0, pool: 100, n_init: 10}` (`pool=100` is intentionally outside the tuner space).
- **Objective (minimize)** = `(mean_best_energy, mean_evals_to_target)` over seeds on LJ-38; `evals_to_target` uses the known minimum `atomica.plot.KNOWN_MINIMA[38] = -173.928427`, and a miss counts as `budget`.
- **Safety:** every proposal passes `validate_params` (clamp `k_acq`, snap `pool`/`n_init`, reject unparseable) before use; an irreparable proposal makes the round fall back to a random draw, recorded `fallback: true`. The LLM output is parsed as data, never executed.
- **Credentials are the user's.** The `anthropic` client reads `ANTHROPIC_API_KEY` or an `ant auth login` profile; no key in code. **Tests must never hit the real API** — use a fake/injected client or a fake proposer.
- **Model:** default `claude-sonnet-5` (a module constant `MODEL`, trivially switchable to `claude-opus-5`).
- Reproducibility via `numpy.random.default_rng(seed)` for the random paths (the LLM path is inherently stochastic — see the deliverable).

---

### Task 1: `strategist.py` — parameter space + validation

**Files:**
- Create: `atomica/strategist.py`
- Test: `tests/test_strategist.py`

**Interfaces:**
- Produces: `PARAM_SPACE`, `DEFAULT_PARAMS`, `validate_params(raw: dict) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_strategist.py
import pytest
from atomica.strategist import validate_params, PARAM_SPACE, DEFAULT_PARAMS

def test_clamps_and_snaps():
    p = validate_params({"k_acq": 9.9, "pool": 70, "n_init": 100})
    assert p["k_acq"] == 3.0           # clamped to [0,3]
    assert p["pool"] == 80             # snapped to nearest of {40,80,160}
    assert p["n_init"] == 20           # snapped to nearest of {5,10,20}

def test_low_clamp():
    assert validate_params({"k_acq": -5, "pool": 40, "n_init": 5})["k_acq"] == 0.0

def test_rejects_unparseable():
    for bad in [{"k_acq": "x", "pool": 40, "n_init": 5}, {"pool": 40}, "nope", None]:
        with pytest.raises(ValueError):
            validate_params(bad)

def test_default_params_shape():
    assert DEFAULT_PARAMS == {"k_acq": 1.0, "pool": 100, "n_init": 10}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_strategist.py -v`
Expected: FAIL with `ModuleNotFoundError: atomica.strategist`.

- [ ] **Step 3: Write minimal implementation**

```python
# atomica/strategist.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_strategist.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/strategist.py tests/test_strategist.py
git commit -m "feat: P3 parameter space and validation"
```

---

### Task 2: `strategist.py` — `score_params`

**Files:**
- Modify: `atomica/strategist.py`
- Test: `tests/test_strategist.py` (add cases)

**Interfaces:**
- Consumes: `atomica.search.active_learning_search`, `atomica.potential.relax`, `atomica.plot.KNOWN_MINIMA`, `atomica.plot.evals_to_target`.
- Produces: `score_params(params, tune_seeds, budget, n=38) -> {"mean_best", "mean_evals", "params"}`.

- [ ] **Step 1: Write the failing test** (tiny budget/seeds so it's fast)

```python
# add to tests/test_strategist.py
from atomica.strategist import score_params

def test_score_params_returns_finite_summary():
    s = score_params({"k_acq": 1.0, "pool": 40, "n_init": 5}, tune_seeds=[0], budget=12, n=38)
    assert set(s) == {"mean_best", "mean_evals", "params"}
    assert s["mean_best"] < 0                 # LJ energies are negative
    assert 0 < s["mean_evals"] <= 12          # evals-to-target within budget (miss counts as budget)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_strategist.py -k score -v`
Expected: FAIL with ImportError for `score_params`.

- [ ] **Step 3: Write minimal implementation** (append to `atomica/strategist.py`)

```python
import numpy as np
from atomica.potential import relax
from atomica.search import active_learning_search
from atomica.plot import KNOWN_MINIMA, evals_to_target

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_strategist.py -k score -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/strategist.py tests/test_strategist.py
git commit -m "feat: P3 score_params objective over LJ-38"
```

---

### Task 3: `strategist.py` — `random_proposer` + shared `tune` loop

**Files:**
- Modify: `atomica/strategist.py`
- Test: `tests/test_strategist.py` (add cases)

**Interfaces:**
- Produces:
  - `random_proposer(rng) -> callable(history) -> dict`
  - `tune(proposer, rounds, tune_seeds, budget, seed=0, n=38) -> (best_params, trace)` where `trace` is a list of per-round dicts (`mean_best`, `mean_evals`, `params`, `round`, `fallback`).

- [ ] **Step 1: Write the failing tests** (use a fake proposer — no search cost when scoring is stubbed? here we run tiny real scores)

```python
# add to tests/test_strategist.py
import numpy as np
from atomica.strategist import random_proposer, tune, PARAM_SPACE

def test_random_proposer_in_space():
    rng = np.random.default_rng(0)
    propose = random_proposer(rng)
    for _ in range(20):
        p = propose([])
        assert 0.0 <= p["k_acq"] <= 3.0
        assert p["pool"] in PARAM_SPACE["pool"]
        assert p["n_init"] in PARAM_SPACE["n_init"]

def test_tune_trace_and_best():
    # scripted proposer: two valid param sets, second is the "obviously good" one on tiny budget
    seq = [{"k_acq": 0.0, "pool": 40, "n_init": 5}, {"k_acq": 1.0, "pool": 80, "n_init": 10}]
    proposer = lambda history: seq[len(history) % len(seq)]
    best, trace = tune(proposer, rounds=2, tune_seeds=[0], budget=12, seed=0)
    assert len(trace) == 2
    assert all(not t["fallback"] for t in trace)
    # best is the argmin of (mean_best, mean_evals) across the trace
    argmin = min(trace, key=lambda t: (t["mean_best"], t["mean_evals"]))["params"]
    assert best == argmin

def test_tune_falls_back_on_garbage():
    proposer = lambda history: {"junk": 1}          # unparseable every round
    best, trace = tune(proposer, rounds=2, tune_seeds=[0], budget=12, seed=1)
    assert all(t["fallback"] for t in trace)         # each round fell back to a random draw
    assert best["pool"] in PARAM_SPACE["pool"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_strategist.py -k "random_proposer or tune" -v`
Expected: FAIL with ImportError for `random_proposer`.

- [ ] **Step 3: Write minimal implementation** (append to `atomica/strategist.py`)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_strategist.py -k "random_proposer or tune" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/strategist.py tests/test_strategist.py
git commit -m "feat: P3 random proposer and shared tune loop with fallback"
```

---

### Task 4: `strategist.py` — `llm_proposer` (Anthropic structured output)

**Files:**
- Modify: `atomica/strategist.py`, `requirements.txt` (add `anthropic`)
- Test: `tests/test_strategist.py` (add cases)

**REQUIRED:** Before writing this task, read the `claude-api` skill's `python/claude-api/tool-use.md` for the exact current structured-output / strict-tool surface. Do not guess the SDK shape.

**Interfaces:**
- Produces:
  - `MODEL = "claude-sonnet-5"`
  - `build_prompt(history) -> str` — a tiny prompt summarizing the param space + `(params → mean_best, mean_evals)` history.
  - `llm_proposer(client, model=MODEL) -> callable(history) -> dict` — one structured Claude call returning raw `{k_acq, pool, n_init}` (validated by the caller `tune`).

**Design for testability:** `llm_proposer` takes an **injected `client`**. Use a single strict tool named `propose_params` with schema `{k_acq: number, pool: integer, n_init: integer}` and `tool_choice` forcing that tool; extract the params from the returned `tool_use` block's `.input`. Tests pass a **fake client** whose `messages.create(...)` returns a canned object shaped like a `tool_use` response — no network, no key.

- [ ] **Step 1: Add dependency**

Append `anthropic` to `requirements.txt`, then `python3 -m pip install anthropic`.

- [ ] **Step 2: Write the failing tests** (fake client — never hits the API)

```python
# add to tests/test_strategist.py
from atomica.strategist import llm_proposer, build_prompt, MODEL

class _FakeBlock:
    def __init__(self, inp): self.type = "tool_use"; self.name = "propose_params"; self.input = inp
class _FakeResp:
    def __init__(self, inp): self.content = [_FakeBlock(inp)]
class _FakeMessages:
    def __init__(self, inp): self._inp = inp; self.last_kwargs = None
    def create(self, **kwargs): self.last_kwargs = kwargs; return _FakeResp(self._inp)
class _FakeClient:
    def __init__(self, inp): self.messages = _FakeMessages(inp)

def test_llm_proposer_extracts_params():
    client = _FakeClient({"k_acq": 2.0, "pool": 80, "n_init": 10})
    propose = llm_proposer(client)
    raw = propose([{"params": {"k_acq": 1.0, "pool": 40, "n_init": 5},
                    "mean_best": -170.0, "mean_evals": 100}])
    assert raw == {"k_acq": 2.0, "pool": 80, "n_init": 10}
    assert client.messages.last_kwargs["model"] == MODEL       # used the configured model

def test_build_prompt_mentions_space_and_history():
    txt = build_prompt([{"params": {"k_acq": 1.0, "pool": 40, "n_init": 5},
                         "mean_best": -170.0, "mean_evals": 100}])
    assert "k_acq" in txt and "pool" in txt and "-170" in txt
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_strategist.py -k "llm_proposer or build_prompt" -v`
Expected: FAIL with ImportError.

- [ ] **Step 4: Write minimal implementation** (append to `atomica/strategist.py`; adjust the exact `client.messages.create` call to match the `claude-api` skill's documented strict-tool surface)

```python
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
        resp = client.messages.create(
            model=model, max_tokens=512,
            tools=[_TOOL], tool_choice={"type": "tool", "name": "propose_params"},
            messages=[{"role": "user", "content": build_prompt(history)}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "propose_params":
                return block.input
        raise ValueError("no propose_params tool_use in response")
    return propose
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_strategist.py -k "llm_proposer or build_prompt" -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add atomica/strategist.py tests/test_strategist.py requirements.txt
git commit -m "feat: P3 LLM proposer via Anthropic strict-tool structured output"
```

---

### Task 5: `strategist.py` `compare` + `run_tune.py` CLI

**Files:**
- Modify: `atomica/strategist.py`
- Create: `atomica/run_tune.py`
- Test: `tests/test_run_tune.py`

**Interfaces:**
- Produces:
  - `compare(best_by_tuner: dict, eval_seeds, budget, n=38) -> dict` — scores each named param set (plus `default`) on the eval-seeds; returns `{name: score_dict}`.
  - `run_tune.main(argv=None) -> None` — runs random tuning (and LLM tuning if a client is available), compares on held-out eval-seeds, writes `results/tune_report.json`.

- [ ] **Step 1: Write the failing test** (fake proposer via monkeypatch — no API)

```python
# tests/test_run_tune.py
import json
from atomica import run_tune

def test_cli_smoke_random_only(tmp_path, monkeypatch):
    # Force the LLM path off so the smoke test needs no credential.
    monkeypatch.setattr(run_tune, "make_llm_proposer", lambda model: None)
    run_tune.main(["--rounds", "2", "--tune-seeds", "1", "--eval-seeds", "2",
                   "--budget", "12", "--trajectories", "1", "--out", str(tmp_path)])
    report = json.loads((tmp_path / "tune_report.json").read_text())
    assert "default" in report["comparison"]
    assert "random" in report["comparison"]
    assert "llm" not in report["comparison"]        # skipped, no client
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_run_tune.py -v`
Expected: FAIL with `ModuleNotFoundError: atomica.run_tune`.

- [ ] **Step 3: Write minimal implementations**

Append to `atomica/strategist.py`:

```python
def compare(best_by_tuner, eval_seeds, budget, n=38):
    everyone = {**best_by_tuner, "default": DEFAULT_PARAMS}
    return {name: score_params(params, eval_seeds, budget, n) for name, params in everyone.items()}
```

Create `atomica/run_tune.py`:

```python
import argparse, json
from pathlib import Path
from atomica.strategist import tune, compare, random_proposer, llm_proposer, MODEL
import numpy as np

def make_llm_proposer(model):
    """Return an llm_proposer bound to a real client, or None if no credential/SDK."""
    try:
        import anthropic
        return llm_proposer(anthropic.Anthropic(), model=model)
    except Exception as e:                     # missing key/sdk -> skip the LLM arm
        print(f"[run_tune] LLM arm disabled: {e}")
        return None

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
        best["llm"] = _best_over_trajectories(
            lambda t: llm, a.trajectories, a.rounds, tune_seeds, a.budget)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_run_tune.py -v` then the full suite `python3 -m pytest -q`.
Expected: PASS (Slice 1 + P2 + P3 all green).

- [ ] **Step 5: Commit**

```bash
git add atomica/strategist.py atomica/run_tune.py tests/test_run_tune.py
git commit -m "feat: P3 compare + run_tune CLI (LLM arm auto-skips without a credential)"
```

---

### Task 6: Deliverable run + README

**Files:**
- Modify: `README.md`

**Interfaces:** none (produces the result + docs).

- [ ] **Step 1: Check for a credential**

Run: `ant auth status 2>&1 || echo "no ant profile"` and check `echo "${ANTHROPIC_API_KEY:+set}"`.
- If a credential is available → the full LLM-vs-random run is possible (Step 2).
- If **not** available → run the random-vs-default portion (no key needed) and note in the README that the LLM arm requires the user's own `ANTHROPIC_API_KEY` / `ant auth login`. Do not fabricate LLM numbers.

- [ ] **Step 2: Run the deliverable**

```bash
python3 -m atomica.run_tune --rounds 6 --tune-seeds 2 --eval-seeds 5 --budget 120 --trajectories 3 --out results
```

This makes a handful of small `claude-sonnet-5` calls (if the LLM arm is on) plus LJ-38 searches (cheap, toy potential). Read `results/tune_report.json`: on the held-out eval-seeds, compare `llm` vs `random` vs `default` `mean_best` / `mean_evals`.

- [ ] **Step 3: Write the README "P3" section**

Add a "P3 — LLM strategist tunes the search" section to `README.md`: what it does (LLM proposes bounded hyperparameters as validated JSON, never touches physics), how to run (`python3 -m atomica.run_tune`, note the `ANTHROPIC_API_KEY` requirement), and the result honestly — does LLM tuning beat random tuning and the default on LJ-38? Include the caveat that the LLM is stochastic (a few trajectories) and that a null result is valid. Point to the P3 spec.

- [ ] **Step 4: Run the full test suite**

Run: `python3 -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add README.md results/tune_report.json
git commit -m "feat: P3 deliverable run and README (LLM vs random hyperparameter tuning)"
```

---

## Self-Review

**1. Spec coverage:**
- Bounded param space + validation (spec §2/§3) → Task 1. ✅
- `score_params` objective on LJ-38 vs `KNOWN_MINIMA[38]` (spec §2/§4) → Task 2. ✅
- Shared `tune` loop + random proposer + fallback-on-garbage (spec §3/§4) → Task 3. ✅
- `llm_proposer` as validated structured output, injected client, no code execution, model constant (spec §3/§4) → Task 4. ✅
- `compare` + CLI + LLM-arm auto-skips without a credential (spec §3/§7) → Task 5. ✅
- Deliverable (LLM vs random vs default) + honest caveats + credential-gated run (spec §6/§7) → Task 6. ✅
- Tests never hit the real API (fake client / fake proposer / monkeypatched LLM arm) (spec §5) → Tasks 4, 5. ✅
- `anthropic` dependency, credential is the user's (spec §7) → Task 4, Task 6 Step 1. ✅

**2. Placeholder scan:** No "TBD"/"handle edge cases"/placeholder code. Task 4 explicitly requires confirming the exact Anthropic strict-tool surface from the `claude-api` skill before writing — the shown code is the concrete target, adjusted only if the SDK surface differs.

**3. Type consistency:** `validate_params(raw)->dict{k_acq,pool,n_init}` used in Tasks 1,3,5. `score_params(params, seeds, budget, n)->{mean_best,mean_evals,params}` used in Tasks 2,3,5. `tune(proposer, rounds, tune_seeds, budget, seed, n)->(best_params, trace)` in Tasks 3,5. `proposer` protocol `propose(history)->raw dict` for both random and LLM (Tasks 3,4). `compare(best_by_tuner, eval_seeds, budget, n)->{name:score}` in Task 5. `MODEL` constant in Tasks 4,5. `make_llm_proposer(model)->proposer|None` defined and monkeypatched in Task 5.
