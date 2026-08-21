# ⚛️ ATOMICA

> An AI-guided scientific-discovery loop for atomic systems — where every claim is backed by measurable, reproducible computation, never by the AI's say-so.

![license](https://img.shields.io/badge/license-MIT-green)
![python](https://img.shields.io/badge/python-3.13-blue)
![tests](https://img.shields.io/badge/tests-66%20passing-brightgreen)

ATOMICA asks a single, cheat-proof question and answers it with numbers you can check: **under an equal compute budget, can AI/ML-guided search find good atomic configurations faster than classical baselines?** It is built in slices — each one a small, self-contained, measurable result.

---

## ✨ Highlights

- 🧪 **Slice 1 — toy potential:** AI-guided search **loses** to a genetic baseline. An honest negative result.
- 🔬 **P2 — real ML potential (MACE):** on a Cu-Au alloy-ordering problem, AI-guided search **wins** — fewer evaluations, higher hit rate.
- 🎯 Every result is validated against **ground truth** (known global minima / brute force), so "it worked" is never a matter of opinion.
- 🤖 **P3 — LLM enters the loop, bounded:** it proposes hyperparameters as validated JSON, never physics; see below for the honestly-reported result.
- 🧑‍⚖️ **P4 — LLM as critic, not oracle:** it names a suspected confounder as validated JSON; a harness-side stratified control, not the LLM, decides accept/reject.

---

## 🧭 The idea

The long-term vision is a system that reads scientific literature, proposes hypotheses about atomic structures, runs reproducible experiments to test them, criticizes its own conclusions, and decides what to investigate next — with a human as supervisor, not operator.

That full vision is **not** what this repo is yet. It's deliberately broken into slices, each producing a working, measurable result on its own.

---

## 🧪 Slice 1 — Does AI-guided search beat classical baselines?

**Problem:** Lennard-Jones cluster global-minimum search — place *N* atoms to minimize total LJ energy. Known global minima (Cambridge Cluster Database) give the ground truth. Runs on a laptop, reduced units (ε = σ = 1), no LLM.

Three strategies compete under an equal budget of local relaxations, differing only in *how they use past evaluations*:

| Method | Memory of past evaluations |
|--------|----------------------------|
| **Random** | None |
| **Genetic** | Implicit — population + cut-and-splice crossover + selection |
| **Active-learning** | Explicit — a RandomForest surrogate over a distance-histogram descriptor picks where to look next |

**Results** (`N ∈ {13, 38}`, 5 seeds, budget = 200 relaxations):

- **LJ-13** — every method finds the global minimum (−44.326801); active-learning ties on quality but converges slower (~165 relaxations vs ~25–50).
- **LJ-38** — a hard funnel; nobody reaches the true minimum (−173.928427) within budget:

| Method | mean best | best of 5 | gap to true min |
|--------|----------:|----------:|----------------:|
| 🥇 Genetic | −172.11 | −173.13 | 1.82 |
| Active-learning | −170.49 | −171.16 | 3.44 |
| Random | −169.71 | −170.21 | 4.21 |

![LJ-13 convergence](results/convergence_N13.png)
![LJ-38 convergence](results/convergence_N38.png)

**❌ Verdict:** active-learning, as specified, does **not** beat the baselines here — a real, reported-as-is negative result. Likely causes: the distance-histogram descriptor is lossy and the acquisition is untuned. Fixing that is future work, not a Slice 1 goal.

---

## 🔬 P2 — Cu-Au alloy ordering, with a real ML potential

**Problem:** swap the toy potential for **MACE-MP-0** (`mace-torch`) and ask the same question on a real materials problem — on a fixed 12-site Cu-Au FCC lattice (6 Au, 6 Cu, only the *arrangement* varies), find the lowest-energy ordering under an equal MACE-evaluation budget.

Because there are only C(12,6) = 924 configurations, **ground truth is brute-forced** with the real potential — every ordering evaluated once and cached.

```bash
python3 -m atomica.run_alloy --budget 100 --seeds 5 --out results/alloy
```

**Result** — brute-forced global minimum **−44.36847 eV**, config `[0, 1, 4, 5, 8, 9]`:

| Method | mean best | success rate | mean evals-to-target |
|--------|----------:|:------------:|---------------------:|
| 🥇 Active-learning | −44.36847 | 5/5 (100%) | **17.2** |
| Genetic | −44.36847 | 5/5 (100%) | 26.8 |
| Random | −44.29604 | 2/5 (40%) | 28.5 |

![Cu-Au convergence](results/convergence_N12.png)

**✅ Verdict:** active-learning **wins** — reaches the true ground state every seed, in the fewest evaluations. The opposite of Slice 1: on a real potential with a small, cheat-proof search space, the surrogate delivers a genuine, measured speedup.

> 🧲 **Physics check:** the ground state is a pure **L1₀ layering** (alternating Cu / Au (100) planes) — exactly the real CuAu-I ordered intermetallic. A reassuring sanity check on the MACE potential.

---

## 🤖 P3 — Does an LLM strategist tune the search better than random?

**Problem:** Slice 1 flagged that active-learning's acquisition hyperparameters (`k_acq`, `pool`,
`n_init`) were **untuned**. P3 asks the natural follow-up: can an LLM, reading `(params, result)`
history and proposing the next hyperparameters as validated structured JSON, tune them better than
random guessing? The LLM's entire action space is three bounded numbers — it never touches the
physics, never runs code, never chooses the problem. Every proposal is clamped/snapped by
`validate_params`, and a malformed proposal falls back to a random draw for that round.

Three tuners compete on **LJ-38** under an equal budget of tuning rounds, then the best parameters
each found are scored on **held-out eval-seeds** (disjoint from tuning) against the fixed Slice-1
default:

| Tuner | How it picks the next hyperparameters |
|-------|----------------------------------------|
| **Default** | Fixed Slice-1 values (`k_acq=1.0, pool=100, n_init=10`) — no tuning |
| **Random** | Uniform draw each round |
| **LLM** | Reads tuning history, proposes next params as a validated tool call |

```bash
python3 -m atomica.run_tune --rounds 6 --tune-seeds 2 --eval-seeds 5 --budget 120 --trajectories 3 --out results
```

The LLM arm needs **your own** `ANTHROPIC_API_KEY` (or an `ant auth login` profile) — the code never
contains a key. Without one, the CLI still runs the random-vs-default comparison and prints
`[run_tune] LLM arm disabled: ...` instead of failing.

**Result** (held-out eval-seeds, budget = 120, LLM arm requires a credential — see below):

| Tuner | mean best | mean evals-to-target |
|-------|----------:|----------------------:|
| Random | −170.998 | 120.0 (never hit target) |
| Default | −170.714 | 120.0 (never hit target) |

Neither tuner reaches the true LJ-38 minimum (−173.928427) within this budget, but **random-tuned
parameters (`k_acq=2.79, pool=40, n_init=10`) beat the fixed default** on mean best-energy — a small,
real edge from just *searching* the hyperparameter space at all, even without an LLM.

**⚠️ Honest note:** this environment has no `ANTHROPIC_API_KEY` and no `ant` profile configured, so
the **LLM arm was not run here** (no credential) — the table above is random-vs-default only. Running
the CLI regenerates `results/tune_report.json` with the `random` and `default` arms only (the CLI
printed `[run_tune] LLM arm disabled: ...` and continued). The LLM-vs-random comparison is left for
you to run with your own key; a null result (LLM ≈ random) is a valid outcome, just as Slice 1's
negative result was.

**Caveat:** the LLM is a stochastic proposer sampled over a handful of trajectories (`--trajectories`)
— re-running can shift its result. See the design spec:
[`docs/superpowers/specs/2026-08-19-atomica-p3-llm-tuner-design.md`](docs/superpowers/specs/2026-08-19-atomica-p3-llm-tuner-design.md).

---

## 🧑‍⚖️ P4 — Does an LLM critic catch false claims better than random or none?

**Problem:** reuse P2's Cu-Au world (the same 924-config brute-forced ground truth). A deterministic
"scientist" draws a confounded, biased 40-config sample and emits a claim of the form *"feature X
drives energy"* — some claims are TRUE, some are FALSE (the apparent effect is really driven by a
different, correlated feature). An LLM critic reads only the claim and the sample, never the physics,
and returns a validated strict-tool JSON verdict: `supported`, or `confounded` (naming which feature
it thinks is the confounder). A within-sample **stratified control** — bin the sample on the named
confounder, re-check the sign of the target's effect within each bin — decides accept/reject by
sign-flip: harness logic, not the LLM's say-so. Everything is checked against the same P2 ground
truth, so a claim's TRUE/FALSE label is never a matter of opinion.

Three critics compete on the same 60 labeled claims:

| Critic | How it decides |
|--------|-----------------|
| **none** | Accepts every claim (no critic at all) |
| **random** | Names a random confounder and applies the same stratified control |
| **llm** | Reads the claim + sample, names a confounder via validated tool call, same control |

```bash
python3 -m atomica.run_critic --n-claims 60 --n 40 --strength 2.0 --seed 0 --out results
```

The **measured question**: on 60 claims, does the LLM critic's false-discovery rate (FDR — fraction
of accepted claims that are actually FALSE) fall below `none` and `random`, at matched true-claim
retention? The LLM arm needs **your own** `ANTHROPIC_API_KEY` (or an `ant auth login` profile) — the
code never contains a key. Without one, the CLI still runs `none` + `random` and prints
`[run_critic] LLM arm disabled: ...` instead of failing.

**Result** (60 claims, `n=40`, `strength=2.0`, seed 0):

| Critic | FDR | retention |
|--------|----:|----------:|
| none | 0.233 | 1.000 |
| random | 0.179 | 1.000 |
| llm | — not run — | — not run — |

**⚠️ Honest note:** this environment has no `ANTHROPIC_API_KEY` and no `ant` profile configured, so
the CLI printed `[run_critic] LLM arm disabled: "Could not resolve authentication method. Expected
one of api_key, auth_token, or credentials to be set. Or for one of the \`X-Api-Key\` or
\`Authorization\` headers to be explicitly omitted"` and `results/critic_report.json` was written with
only the `none` and `random` arms. The **LLM-vs-random comparison was not run here** — no LLM numbers
are fabricated. It's left for you to run with your own key; a null result (LLM ≈ random) would be
just as valid as Slice 1's negative result.

**Caveats:** the LLM is a stochastic critic — re-running can shift its verdicts. The confound-fix
ceiling is **below 100%** even in principle: some FALSE claims here are small-sample noise rather than
a genuine confound, so no critic (LLM included) can catch every one via confounder-naming alone. The
world is a small 12-site lattice, so these FDR numbers won't generalize to larger search spaces
as-is. See the design spec:
[`docs/superpowers/specs/2026-08-21-atomica-p4-llm-critic-design.md`](docs/superpowers/specs/2026-08-21-atomica-p4-llm-critic-design.md).

---

## 🚀 Quickstart

```bash
python3 -m pip install -r requirements.txt   # Python 3.13; ase, matplotlib, numpy, scipy, scikit-learn, mace-torch
```

```bash
# Slice 1 — LJ cluster search
python3 -m atomica.run --n 13 38 --budget 200 --seeds 5 --out results/lj

# P2 — Cu-Au alloy ordering (real MACE)
python3 -m atomica.run_alloy --budget 100 --seeds 5 --out results/alloy

# P3 — LLM-vs-random hyperparameter tuning (needs ANTHROPIC_API_KEY for the LLM arm; runs random-vs-default without one)
python3 -m atomica.run_tune --rounds 6 --tune-seeds 2 --eval-seeds 5 --budget 120 --trajectories 3 --out results

# P4 — LLM critic false-discovery benchmark (needs ANTHROPIC_API_KEY for the LLM arm; runs none-vs-random without one)
python3 -m atomica.run_critic --n-claims 60 --n 40 --strength 2.0 --seed 0 --out results

# Tests
python3 -m pytest -q
```

> Tip: give the two CLIs separate `--out` directories (as above); both default to `results/`, so sharing it makes each regenerate the other's figures.

---

## 🗂️ Project layout

```
atomica/
├── potential.py      # ASE Lennard-Jones calculator + local relaxation (swap point for real ML potentials)
├── descriptor.py     # pairwise-distance histogram (permutation/rotation/translation invariant)
├── search.py         # random / genetic / active-learning — shared signature, shared budget
├── benchmark.py      # runs method × seed × N, logs reproducible per-run JSON
├── plot.py           # convergence curves + success-rate / evals-to-target metrics (shared)
├── run.py            # CLI — Slice 1 (LJ clusters)
├── alloy.py          # MACE-MP-0 evaluate on a fixed Cu-Au FCC lattice, SRO descriptor, brute-force ground truth
├── alloy_search.py   # random / genetic / active-learning over Au/Cu orderings (composition-preserving)
├── run_alloy.py      # CLI — P2 (Cu-Au alloy ordering)
├── strategist.py     # bounded param space, validation, tune/compare loop, LLM proposer (structured tool call)
├── run_tune.py       # CLI — P3 (LLM-vs-random hyperparameter tuning)
├── critic_world.py   # P4: Cu-Au features, stratified controlled-effect estimator, biased scientist + claims
├── critic.py         # P4: strict-tool critique validation, stratified sign-flip control, arm scoring
└── run_critic.py     # CLI — P4 (LLM critic false-discovery benchmark)
```

Design specs and implementation plans live under [`docs/superpowers/`](docs/superpowers/).

---

## 🗺️ Roadmap

The computational core comes first; LLM work is deliberately last — it has the lowest measurability and the highest risk of looking impressive while meaning nothing.

| Phase | Adds | Status |
|-------|------|:------:|
| **P1** | Search benchmark on a toy LJ potential | ✅ done |
| **P2** | Real ML potential (MACE-MP-0) on a Cu-Au alloy-ordering problem | ✅ done |
| **P3** | An LLM *strategist* that reads results and proposes the next experiment (never touches the physics) | ✅ done |
| **P4** | An LLM *critic* proposing control / falsification experiments | ✅ done |
| **P5** | A literature agent (paper → gaps → hypotheses) feeding P3 | 🔒 planned |

---

## 📄 License

[MIT](LICENSE) © 2026 Chayanun Yuvanaboon
