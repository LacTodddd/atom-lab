# ATOMICA

**An AI-guided scientific-discovery loop for atomic systems.**

The long-term vision: a system that reads scientific literature, proposes
hypotheses about atomic structures, runs reproducible computational experiments
to test them, criticizes its own conclusions, and decides what to investigate
next — with a human as supervisor rather than operator. The guiding principle is
that the AI helps *decide which question to test next*, while every scientific
claim is backed by measurable, reproducible computation, never by the AI's
say-so.

That full vision is deliberately **not** what this repository builds yet. It is
broken into slices, each of which produces a working, measurable result on its
own. This repo is **Slice 1**.

---

## Slice 1 — Does AI-guided search beat classical baselines?

Slice 1 builds the measurable core first, with **no LLM involved**. It asks one
crisp, cheat-proof question:

> Under an equal compute budget, can an AI/ML-guided search find low-energy
> atomic configurations faster than conventional baselines?

The test problem is **Lennard-Jones (LJ) cluster global-minimum search**: place
*N* atoms so their total LJ energy is as low as possible. LJ clusters are a
classic optimization benchmark with **known global minima** (Cambridge Cluster
Database), so we have ground truth to validate against. Everything runs on a
laptop in reduced units (ε = σ = 1).

Three search strategies compete under the **same budget of local relaxations**
(each relaxation = one budget unit; the surrogate's predictions are free). They
differ in one axis — *how much they use past evaluations*:

| Method | How it uses memory of past evaluations |
|--------|----------------------------------------|
| **Random** | None — sample, relax, keep the best |
| **Genetic** | Implicit — a population plus cut-and-splice crossover + selection |
| **Active-learning** | Explicit — a RandomForest surrogate learns the energy landscape and proposes where to look next (lower-confidence-bound acquisition) |

Structures are fed to the surrogate through a permutation/rotation/translation-
invariant **pairwise-distance histogram** descriptor.

---

## Results

Full sweep: `N ∈ {13, 38}`, 3 methods, 5 seeds, budget = 200 relaxations.

**LJ-13** — every method finds the exact global minimum (−44.326801). The
differentiator is speed: Random and Genetic get there in ~25–50 relaxations;
Active-learning ties on final quality but is markedly slower (~165).

**LJ-38** — a genuine funnel problem; no method reached the true minimum
(−173.928427) within budget.

| Method | mean best energy | best of 5 seeds | mean gap to true min |
|--------|-----------------:|----------------:|---------------------:|
| **Genetic** | −172.11 | −173.13 | 1.82 |
| Active-learning | −170.49 | −171.16 | 3.44 |
| Random | −169.71 | −170.21 | 4.21 |

![LJ-13 convergence](results/convergence_N13.png)
![LJ-38 convergence](results/convergence_N38.png)

**Verdict: the active-learning method, as specified, does *not* beat the
baselines.** It loses to Genetic at N=38 and merely ties (while converging
slower) at N=13.

This is a real, reported-as-is negative result — which is the point. The
experiment was designed to be measurable and cheat-proof, and the honest answer
under these settings is "not yet." Plausible reasons: the distance-histogram
descriptor is lossy (distinct structures can map to similar histograms), and the
acquisition parameters (`k_acq`, candidate-pool size, initial sample count) are
untuned. Making the surrogate competitive — via a richer descriptor, tuning, or
a surrogate-screened basin-hopping scheme — is future work, not a Slice 1 goal.

---

## P2 — Cu-Au alloy ordering (real MACE potential)

P2 swaps the toy LJ potential for a real ML interatomic potential
(**MACE-MP-0**, `mace-torch`) and asks the same cheat-proof question on a real
materials problem: on a fixed 12-site Cu-Au FCC lattice (6 Au, 6 Cu —
composition fixed, only the *arrangement* varies), can AI-guided search find
the lowest-energy Cu/Au ordering faster than classical baselines, under equal
MACE-evaluation budget?

Because the site count (C(12,6) = 924 configurations) is small enough,
**ground truth is brute-forced** with the real MACE potential rather than
taken from a literature table — every one of the 924 orderings is evaluated
once and cached (`results/alloy_ground_truth.json`).

Run it:

```bash
python3 -m atomica.run_alloy --budget 100 --seeds 5 --out results
```

### Result

Brute-forced global minimum: **−44.36847 eV**, config (Au on sites)
`[0, 1, 4, 5, 8, 9]` out of 924 evaluated configurations.

| Method | mean best (5 seeds) | best of 5 seeds | success rate (hit ground state) | mean evals-to-target |
|--------|---------------------:|-----------------:|:--------------------------------:|----------------------:|
| **Active-learning** | −44.36847 | −44.36847 | 5/5 (100%) | **17.2** |
| **Genetic** | −44.36847 | −44.36847 | 5/5 (100%) | 26.8 |
| Random | −44.29604 | −44.36847 | 2/5 (40%) | 28.5 (of the 2 that hit it) |

![Cu-Au N12 convergence](results/convergence_N12.png)

**Verdict: active-learning wins here.** Under equal budget it reached the
brute-forced global minimum in every seed, and did so in the fewest MACE
evaluations on average (17.2 vs Genetic's 26.8). Genetic also converges
reliably (5/5) but slower. Random only reaches the true ground state in 2 of 5
seeds. This is the opposite of the Slice-1 LJ-38 result (where active-learning
lost) — on this real-potential, small-search-space problem, the RandomForest
surrogate over the SRO descriptor gives a genuine, honestly-measured speedup.

Bonus physics note: the ground-state ordering is a pure **L1₀-type layering**
— every Au atom sits on one (100) plane (x = 0) and every Cu atom on the
adjacent (100) plane (x = a/2), i.e. alternating pure Cu/Au planes rather than
a mixed arrangement. That matches the real CuAu-I ordered intermetallic
structure, which is a reassuring physical sanity check on the MACE potential.

See [`docs/superpowers/specs/2026-08-17-atomica-p2-alloy-ordering-design.md`](docs/superpowers/specs/2026-08-17-atomica-p2-alloy-ordering-design.md)
for the full P2 design spec.

## Install

```bash
python3 -m pip install -r requirements.txt
```

Requires Python 3.13. Dependencies: `ase`, `matplotlib`, `numpy`, `scipy`,
`scikit-learn`.

## Run

```bash
python3 -m atomica.run --n 13 38 --budget 200 --seeds 5
```

Flags: `--n` (one or more cluster sizes), `--budget` (relaxations per method),
`--seeds` (repeats, uses seeds `0..N-1`), `--methods` (subset of
`random genetic active`), `--out` (output directory). Writes
`results/convergence_N{n}.png` plus per-run JSON.

## Tests

```bash
python3 -m pytest -q
```

The most important test relaxes a known LJ-13 icosahedron and asserts it reaches
the reference energy — this validates that the potential, the relaxation, and
the harness are all correct.

---

## Project layout

```
atomica/
├── potential.py     # ASE Lennard-Jones calculator + local relaxation (the swap point for real ML potentials later)
├── descriptor.py    # pairwise-distance histogram (permutation/rotation/translation invariant)
├── search.py        # random / genetic / active_learning — one shared signature, one shared budget
├── benchmark.py     # runs method × seed × N, logs reproducible per-run JSON
├── plot.py          # convergence curves + success-rate / evals-to-target metrics (LJ + alloy, shared)
├── run.py           # CLI entry point (P1 — LJ clusters)
├── alloy.py         # MACE-MP-0 evaluate on a fixed Cu-Au FCC lattice, SRO descriptor, brute-force ground truth
├── alloy_search.py  # random / genetic / active_learning over Au/Cu site orderings (composition-preserving)
└── run_alloy.py     # CLI entry point (P2 — Cu-Au alloy ordering)
```

## Roadmap

Ordered by what de-risks the most and is most measurable. LLM work is
deliberately last — it has the lowest measurability and the highest risk of
looking impressive while meaning nothing, so the trustworthy computational core
comes first.

| Phase | Adds | Deliverable |
|-------|------|-------------|
| **P1 (this repo)** | Search benchmark on a toy LJ potential | Does AI-guided beat the baselines? |
| P2 | A real ML potential (MACE/CHGNet) + a small real problem (vacancy/substitution), via the same `potential` interface | A real-physics result on the same harness |
| P3 | An LLM *strategist* that reads results and proposes the next experiment (never touches the physics) | A semi-autonomous, human-supervised loop |
| P4 | An LLM *critic* proposing control/falsification experiments | Reduced false-discovery rate |
| P5 | A literature agent (paper → gaps → hypotheses) feeding P3 | The full vision |

See [`docs/superpowers/specs/2026-08-13-atomica-slice1-design.md`](docs/superpowers/specs/2026-08-13-atomica-slice1-design.md)
for the full design and rationale, and
[`docs/superpowers/plans/2026-08-13-atomica-slice1-search-benchmark.md`](docs/superpowers/plans/2026-08-13-atomica-slice1-search-benchmark.md)
for the implementation plan.
