# ATOMICA — P3: LLM Strategist as a Hyperparameter Tuner

- **Date:** 2026-08-19
- **Status:** Approved (design), pending implementation plan
- **Builds on:** Slice 1 (`atomica/search.py::active_learning_search`, which already exposes the
  `n_init` / `pool` / `k_acq` knobs) and its honest, ground-truth-validated methodology.
- **Roadmap position:** Phase P3 — the first time an LLM enters the loop.

---

## 1. Context

P1 and P2 answered crisp, ground-truth-checked questions. P3 introduces the LLM — the riskiest
step, because "did the LLM help?" is fuzzy where "did we hit the global minimum?" is not. So P3 is
scoped to a setting where the LLM's contribution is **measurable and cheat-proof**.

Slice 1's headline negative result was that active-learning search, *as specified*, lost to the
genetic baseline on LJ-38 — and we flagged that its acquisition parameters were **untuned**. P3 asks
the natural follow-up:

> Can an LLM, acting as an experiment strategist, tune active-learning's hyperparameters from
> observed results better than random guessing — enough to close (or beat) the gap?

The LLM never touches the physics. It only proposes hyperparameters; the real harness runs the
search and produces the numbers that decide the outcome (the §4 principle from the original plan).

## 2. The measurable claim

The problem is **LJ-38** (maximum headroom: active-learning lost there and was noted under-tuned).

We compare three tuners of `active_learning_search`'s parameters, all under an equal budget of
**R tuning rounds**:

- **Default** — the exact fixed Slice-1 parameters (`k_acq=1.0, pool=100, n_init=10`; `pool=100` is intentionally outside the tuner space — it's the "what Slice 1 shipped" reference). No tuning.
- **Random** — each round draws parameters uniformly from the allowed space (no LLM).
- **LLM** — each round, the LLM sees the history of `(params, result)` and proposes the next
  parameters (validated structured output).

Each proposal is scored by running `active_learning_search` on LJ-38 over **T tune-seeds** and
summarizing convergence (mean best-energy and mean evals-to-target). After R rounds, the
best-scoring parameters from **Random** and from **LLM** are evaluated on **E held-out eval-seeds**
and compared against **Default** on the same eval-seeds.

> **Headline question:** under equal rounds, does the LLM propose better parameters than random
> search? (LLM ≈ random ⇒ the LLM adds nothing — a valid, honestly-reported negative result, exactly
> as in Slice 1.)

**Parameter space (bounded, validated):**

| Param | Space | Type |
|-------|-------|------|
| `k_acq` | `[0.0, 3.0]` | float (LCB weight) |
| `pool` | `{40, 80, 160}` | int (candidate-pool size) |
| `n_init` | `{5, 10, 20}` | int (initial random samples) |

## 3. Safety & the LLM boundary

- The LLM's entire action space is **three bounded parameters** returned as **validated structured
  JSON**. It cannot emit or run code, choose problems, or touch the physics.
- Every proposal is **validated by the harness** before use: `k_acq` clamped to `[0, 3]`; `pool`
  and `n_init` snapped to the nearest allowed value; a malformed/unparseable proposal is rejected
  and that round falls back to a random draw (recorded as such).
- The physics/search is always the real Slice-1 harness. The human supervises; the run is
  reproducible except for the LLM's own stochasticity (see §6).
- **Credentials are the user's:** the SDK reads `ANTHROPIC_API_KEY` or an `ant auth login` profile.
  The code never contains a key; if no credential is configured, the LLM tuner errors out clearly
  and the random/default paths still run.

## 4. Architecture

New module `atomica/strategist.py`, plus a small CLI. Reuses `active_learning_search` unchanged.

### Interfaces

- `PARAM_SPACE` — the bounded space above; `validate_params(raw) -> dict` clamps/snaps and raises on
  irreparable input.
- `score_params(params, tune_seeds, budget) -> dict` — runs `active_learning_search` on LJ-38 for
  each tune-seed, returns `{mean_best, mean_evals_to_target, per_seed}`. This is the shared,
  physics-real objective every tuner is scored against.
- A proposer protocol: `propose(history) -> dict` (raw params). Two implementations:
  - `random_proposer(rng)` — uniform draw from `PARAM_SPACE`.
  - `llm_proposer(model, client)` — one Claude call (Anthropic SDK, structured output) given the
    history and the space; returns validated params, or signals fallback.
- `tune(proposer, rounds, tune_seeds, budget) -> (best_params, trace)` — the shared R-round loop:
  propose → validate → `score_params` → record. Identical for random and LLM (only the proposer
  differs), so the comparison is fair by construction.
- `compare(best_params_by_tuner, default_params, eval_seeds, budget) -> results` — evaluates each
  parameter set on the held-out eval-seeds and returns the comparison table.

### LLM call

- Anthropic Python SDK; model default **`claude-sonnet-5`** (a module constant, trivially switchable
  to `claude-opus-5`), adaptive thinking, low `max_tokens` (the reply is a tiny JSON object).
- **Structured output** so the reply validates to `{k_acq: float, pool: int, n_init: int}` — this is
  the mechanism that keeps the action space constrained. Follow the `claude-api` skill for the exact
  current SDK surface (`messages.parse` / `output_config.format` / strict tool schema) at build time.
- System prompt frames it as an experiment strategist tuning an active-learning search to minimize
  cluster energy in the fewest evaluations; the user message carries the param space and the
  `(params → mean_best, mean_evals)` history so far. No chain-of-code, only the next parameters.

### Data flow

```
tune(random_proposer)  ┐
tune(llm_proposer)     ┤─ each: R rounds of  propose → validate → score_params(LJ-38, tune-seeds)
                       ┘                                                     │
best params (random), best params (llm), default ──▶ compare(eval-seeds) ──▶ results/tune_report.json + a small table/plot
```

## 5. Testing & validation

Tests must **never** hit the real Anthropic API or require a key — use a **fake proposer** (a stub
returning canned/valid params, or a scripted sequence) to exercise the loop.

- **Param validation (key):** `validate_params` clamps `k_acq` out of range, snaps `pool`/`n_init`
  to the nearest allowed value, and rejects an unparseable/missing-field proposal.
- **Tuner loop:** `tune(fake_proposer, ...)` runs R rounds, records a trace of length R, and returns
  a best-params set that is the argmin of the recorded scores.
- **Random proposer** always yields in-space params; seeded → reproducible.
- **`score_params`** returns a finite `mean_best` and a sane `mean_evals_to_target` for a valid
  param set on a tiny budget.
- **Fallback:** a proposer that returns garbage causes the round to fall back to a random draw,
  recorded as `fallback: true` — no crash.
- The real `llm_proposer` is exercised only in the deliverable run (needs a credential).

## 6. Deliverable / honest caveats

Deliverable: `results/tune_report.json` + a short table (and optional convergence-vs-round plot)
answering — on LJ-38 held-out eval-seeds — does **LLM tuning** beat **random tuning** and the
**default**, under equal rounds?

Honest caveats to report as-is:
- The LLM is **stochastic**; a single trajectory is one sample. Run a few independent LLM
  trajectories (and matching random trajectories) and compare distributions, not one lucky run.
- Headroom is real but bounded — active-learning may still not overtake genetic even fully tuned;
  the measured question is specifically **LLM vs random tuning**, and a null result is a result.

## 7. Dependencies & scope

- Add `anthropic` to `requirements.txt`. Credential (`ANTHROPIC_API_KEY` or `ant auth login`) is the
  user's to provide — never committed.
- **In scope:** the strategist module (param space + validation, `score_params`, random + LLM
  proposers, the shared tune loop, compare), a CLI, tests with a fake proposer, one real deliverable
  run.
- **Out of scope (later phases):** the LLM choosing problems/methods (P3 option B), an LLM critic
  (P4), literature ingestion (P5), tuning on the alloy/MACE problem, any autonomous multi-round
  physics loop beyond this bounded tuner.

## 8. Open items to resolve during implementation

- Confirm the exact current Anthropic structured-output call from the `claude-api` skill (SDK method
  + schema shape) before writing the LLM proposer.
- Pick concrete `R` (rounds), `T` (tune-seeds), `E` (eval-seeds), per-run `budget`, and number of
  trajectories to keep the deliverable run to a few minutes and a handful of cheap LLM calls.
- Decide the exact convergence summary fields fed to the LLM (keep the prompt tiny for cost + cache).
