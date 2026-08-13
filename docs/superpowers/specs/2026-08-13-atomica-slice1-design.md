# ATOMICA — Slice 1 Design & Development Roadmap

- **Date:** 2026-08-13
- **Status:** Approved (design), pending implementation plan
- **Source vision:** `ATOM_PROJECT_PLAN.md` (full 27-section vision)
- **This document:** narrows the vision to a first buildable slice and lays out
  the forward roadmap so the slice extends to the full vision without a rewrite.

---

## 1. Context

The original plan (`ATOM_PROJECT_PLAN.md`) describes a full autonomous AI
scientist: literature agent, hypothesis agent, structure/dataset search, atomic
simulation, analysis, critic, persistent memory, and an autonomous loop. That is
a thesis-scale vision, not a first build.

Two problems drove this design:

1. **Scope is ~10× too large for a first build.** Five LLM agents + memory +
   critic + three search strategies + active learning + anomaly detection cannot
   be built or validated in one pass.
2. **"AI" means two different things in the plan.** (a) An **LLM** that reads
   papers, forms hypotheses, and criticizes; and (b) **ML models** (interatomic
   potentials, surrogates) that accelerate search. These are separate systems
   with very different measurability. The one experiment with a crisp,
   cheat-proof metric (§20: Random vs Genetic vs AI-guided search under equal
   budget) needs only (b) — **no LLM at all**.

## 2. Decisions (from design interview)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| Bar | Rigor level | Hobby / portfolio now, upgrade to publishable rigor later | Prove the core before paying for full reproducibility/stats; but keep seeds + reproducible logging from day one because that is cheap now and expensive to retrofit. |
| Heart | LLM vs ML | Both matter; **build ML-discovery first** | If AI-guided search cannot beat baselines on a toy problem, the grand vision is moot. Build the measurable lab first; LLM plugs in later. |
| Potential | Physics engine | **Toy ASE** (`LennardJones`/`EMT`) behind a swappable calculator interface | 8 GB arm64 Mac + Python 3.13, no MACE/CHGNet installed. Toy runs instantly; real ML potentials are heavy and install-fragile. ASE's calculator interface is the standard swap point for MACE/CHGNet later. |
| Problem | First physics task | **LJ cluster global-minimum search**, N=13 (smoke) → N=38 (benchmark) | Known global minima (Cambridge Cluster Database) give ground truth to validate the harness. Instant to evaluate. Directly instantiates §20. |
| AI-guided | Meaning without an LLM | **Active-learning surrogate loop** | Directly instantiates the plan's "learn from previous experiments" (§5.3, §10). Gives a clean, interpretable trichotomy (see below). |
| Budget | Fairness knobs | 200 local relaxations/method, 5 seeds, N=13 then 38 | 1 seed is noise; relaxations (not wall-clock) are the fair unit. |
| Repo | Structure | Minimal ~6-file package, **not** the §16 mega-scaffold | No LLM, memory, or agents in Slice 1, so those directories do not exist yet. |

### The trichotomy (why these three search methods)

Three levels of "using past evaluations", which is what makes the comparison
interpretable:

| Method | How it uses memory |
|--------|--------------------|
| Random | None |
| Genetic | Implicit (population + selection) |
| Active-learning surrogate | Explicit (a model learns the energy landscape) |

## 3. Slice 1 scope

Benchmark: **find the global-minimum geometry of a Lennard-Jones cluster of N
atoms**, comparing Random vs Genetic vs Active-learning search under an equal
budget of local relaxations.

- Potential: ASE `LennardJones` calculator (reduced units, ε=1, σ=1).
- Local relaxation: ASE optimizer (FIRE or BFGS) to the nearest local minimum.
- Budget: 200 local relaxations per method (only actual relaxations count;
  surrogate predictions are free — that is the point of the surrogate).
- Repeats: 5 seeds per (method, N). Same seed ⇒ same initial random pool across
  methods, so the comparison is fair.
- Sizes: N=13 first (validation/smoke), then N=38 (the real contest).
- **No LLM.**

### Explicit non-goals for Slice 1

Real ML potentials, real materials problems (vacancy/diffusion/adsorption),
literature parsing, hypothesis generation, the critic, persistent research
memory, anomaly detection, and the autonomous loop. All deferred to the roadmap.

## 4. Architecture

Minimal package (`atomica/`), ~6 files:

```
atomica/
├── potential.py     # ASE LJ calculator + local relaxation; the swap seam for MACE/CHGNet
├── descriptor.py    # 30-bin pairwise-distance histogram (permutation/rotation/translation invariant)
├── search.py        # random / genetic / active_learning — one shared signature
├── benchmark.py     # loop over method × seed × N; log per-eval history to results/*.json
├── plot.py          # convergence curves, success-rate bar, evals-to-target table
└── run.py           # CLI entry point
```

### Key interfaces (the forward seams)

- `potential.relax(atoms) -> (relaxed_atoms, energy, n_relaxations)`
  The single swap point: replacing the ASE LJ calculator with a MACE/CHGNet
  calculator is a config change, not a rewrite. Budget is counted here.
- `search(n_atoms, budget, seed, potential[, descriptor]) -> history`
  All three strategies share this signature. `history` is a list of
  `(n_relaxations_used, best_energy_so_far, best_structure)`.

### Data flow

```
run.py (CLI) → benchmark.py
                 └─ for method × seed × N:
                      search.py ── relax ─▶ potential.py (ASE LJ; swap → MACE later)
                         │  (active_learning only) └─▶ descriptor.py
                         └─▶ history → results/<method>_N<n>_seed<s>.json
plot.py: results/*.json → convergence curve + success-rate bar + evals-to-target
```

### Method specifics

- **Random:** sample random clusters (atoms in a bounding sphere/box), relax
  each, keep the best. Baseline with zero memory.
- **Genetic:** population of relaxed clusters; cut-and-splice crossover
  (Deaven–Ho, the standard for clusters) + random-displacement mutation;
  select by energy; iterate generations. Implicit memory.
- **Active-learning surrogate:** maintain a dataset of
  `(distance-histogram descriptor → relaxed energy)`. Each round: generate a
  candidate pool (random clusters + perturbations of the current best), predict
  their energy with a `RandomForestRegressor` (uncertainty = std across trees),
  score by a lower-confidence-bound acquisition (`mean − k·std`, since we
  minimize energy), relax the top-k candidates (spending budget), add the new
  (descriptor, energy) pairs, retrain, repeat. Explicit memory.
  RandomForest is chosen over a Gaussian Process to avoid kernel/hyperparameter
  tuning; the surrogate model is a knob, not a commitment.

### Fairness invariant

Budget is measured as **number of local relaxations**. All methods use the same
`potential.relax`, and a given seed produces the same initial random pool for
every method. Surrogate *predictions* over the candidate pool are free; only
*relaxations* are charged. This prevents surrogate overhead from confounding the
comparison.

## 5. Validation & testing

The single most valuable check, plus two supporting ones (assert-based
self-checks, no framework):

1. **Harness validation (the key test):** run search on LJ-13 and assert the
   best energy reaches the known global minimum (icosahedron) within tolerance.
   This proves the potential, the relaxation, and the search plumbing are all
   correct. Reference values (Cambridge Cluster Database, LJ, ε=σ=1) — **confirm
   exact values during implementation** (use the research skill):
   - N=13 ≈ −44.326801
   - N=38 ≈ −173.928427
2. **Descriptor invariance:** a cluster and a randomly rotated + atom-permuted
   copy must produce identical histograms.
3. **Dimer sanity:** a 2-atom LJ system relaxes to the known equilibrium
   separation and energy (−1 in reduced units).

## 6. Metrics / deliverable

- **Convergence curve:** best energy found vs number of relaxations, averaged
  over seeds (mean ± std).
- **Success rate:** fraction of seeds that reach the known global minimum within
  tolerance.
- **Evals-to-target:** relaxations needed to reach the global minimum (or a
  threshold), per method.

The deliverable answer: *under an equal budget, does the active-learning method
beat Random and Genetic?* A negative result is still valid (see §7).

## 7. Expectation setting (honest)

LJ-13 is easy: every method should find the global minimum; the differentiator
is how many relaxations it takes. LJ-38 is a genuine funnel problem — it is
**possible the surrogate does not beat Genetic**, because the histogram
descriptor is lossy. That is a legitimate scientific finding, not a failure, and
it matches the plan's §11 principle that the AI must be willing to falsify
itself. If a stronger AI-guided contender is wanted later, the upgrade is a
surrogate-screened basin hopping (option (ii) from the interview).

## 8. Development roadmap

Ordered by what de-risks the most and what is most measurable — deliberately
**re-sequenced** from the original plan.

| Phase | Adds | Seam used | Deliverable |
|-------|------|-----------|-------------|
| **P1 (now)** | Search benchmark on toy LJ potential | — | Does AI-guided beat the baselines? (plots) |
| **P2** | Real ML potential (MACE/CHGNet) + one small real problem (vacancy/substitution) | `potential` interface + a `problem` abstraction | A real-physics result on the same harness |
| **P3** | LLM *strategist* that reads results/memory and proposes the next experiment config | Calls the existing benchmark harness; the LLM never touches physics (§4) | A semi-autonomous loop, human-supervised |
| **P4** | LLM *critic* proposing control/falsification experiments | Reads results JSON | Reduced false-discovery rate (§11) |
| **P5** | Literature agent (paper → gaps → hypotheses) feeding P3 | — | The full vision |

### Why the re-sequencing

The original plan puts literature analysis and hypothesis generation early
(Phase 2). This design moves all LLM work to the **end**, because it has the
lowest measurability and the highest risk of looking impressive while meaning
nothing. Build the measurable, cheat-proof core first; add LLM reasoning on top
of a lab that already produces trustworthy numbers.

### What keeps future phases cheap (no rewrite)

- **Swappable potential** → P2 drops in MACE/CHGNet as an ASE calculator.
- **A `problem` abstraction** (Slice 1 hardcodes "LJ cluster geometry";
  P2 generalizes to `{generate candidate, evaluate}`) → new physics tasks.
- **Pluggable search strategies** → add basin hopping, CMA-ES, etc.
- **Reproducible results JSON** → grows into §12 persistent research memory.
- **The LLM sits *on top*** as an orchestrator that picks problems/strategies
  and reads results; it calls the same harness and never does physics, exactly
  as §4 requires.

## 9. Open items to resolve during implementation

- Confirm exact LJ global-minimum energies from the Cambridge Cluster Database
  (research skill).
- Confirm `ase` (and `matplotlib` for plots) install cleanly on Python 3.13 /
  arm64; pin versions if needed.
- Choose the ASE local optimizer (FIRE vs BFGS) and convergence criteria.
- Tune the active-learning knobs (candidate-pool size, top-k per round,
  acquisition weight `k`) — but only after the harness is validated on LJ-13.
