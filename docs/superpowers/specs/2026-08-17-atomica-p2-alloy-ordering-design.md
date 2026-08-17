# ATOMICA — P2: Alloy-Ordering Search with a Real ML Potential

- **Date:** 2026-08-17
- **Status:** Approved (design), pending implementation plan
- **Builds on:** Slice 1 (`docs/superpowers/specs/2026-08-13-atomica-slice1-design.md`) — reuses its
  benchmark loop, metrics, and the Random/Genetic/Active-learning trichotomy.
- **Roadmap position:** Phase P2 (real ML potential + a small real problem via the swappable
  potential seam).

---

## 1. Context

Slice 1 answered "does AI-guided search beat baselines?" on a toy Lennard-Jones cluster (answer:
not as specified). P2 repeats the *same measurable contest* on a **real materials problem with a
real ML interatomic potential**, keeping the result cheat-proof.

A feasibility spike (2026-08-17) confirmed the real potential runs comfortably on this machine
(8 GB arm64 Mac, Python 3.13, torch 2.10):

- `mace-torch` 0.3.16 installs and imports; MACE-MP-0 ("small") loads in ~6.5 s (cached thereafter).
- A 32-atom Cu supercell: single-point energy in **0.29 s**, a 20-step relaxation in 2.5 s.
- Peak RSS **~1.0 GB**; energy −4.09 eV/atom (physically sane; relaxation lowered it correctly).

So a real-potential search benchmark of a few hundred single-point evaluations is a few-minute run.

## 2. Problem

**Cu–Au alloy ordering on a fixed FCC lattice.** Given a fixed 12-site FCC supercell at fixed
composition (6 Cu + 6 Au), find the arrangement of Cu/Au on the sites that minimizes the MACE-MP-0
single-point energy. This is a **discrete combinatorial search** over site labelings — the direct
discrete analogue of Slice 1's continuous cluster search.

- **Lattice:** a 1×1×3 supercell of the conventional 4-atom FCC cell (12 sites), lattice constant
  `a = 3.85 Å` (Vegard mean of Cu 3.615 / Au 4.078). Cell and positions are **fixed** — atoms are
  not relaxed; only their species labels change.
- **Composition:** exactly 6 Au + 6 Cu (all operators preserve this).
- **Evaluation:** rigid **single-point** MACE-MP-0 energy of the labeled supercell. One evaluation =
  one budget unit. (Chosen over per-config relaxation: ~10× faster, keeps brute-force ground truth
  feasible, and ordering energetics are dominated by chemistry, not small relaxations.) The single
  best configuration found is relaxed once at the end as a physicality check.

### Ground truth (cheat-proof target)

The configuration space is `C(12,6) = 924`. We **brute-force all 924** with MACE once (~5 min),
cache the results, and take the exact minimum-energy configuration as the search target. This makes
"did the search find the global minimum?" an exact, self-consistent question against the very
potential being searched. **Bonus physics check (reported, not asserted):** whether MACE's global
minimum corresponds to the physically-known Cu-Au ordering (L1₀-like layering at 50/50).

## 3. Decisions

| Topic | Choice | Rationale |
|-------|--------|-----------|
| Potential | MACE-MP-0 "small", CPU, float64 | Spike-confirmed feasible; accurate; float64 recommended for optimization |
| System | Cu-Au, FCC, 12-site (1×1×3 conventional), a=3.85 Å | Small enough to brute-force ground truth; classic ordering alloy |
| Composition | 6 Cu + 6 Au, fixed | Fixed-composition ordering problem |
| Evaluation | Rigid single-point energy | Fast; brute-force feasible; relax only the final best |
| Ground truth | Brute-force all 924 configs once, cache | Exact, cheat-proof, self-consistent target |
| Budget | 100 evaluations per method, 5 seeds | ~11% of the 924-config space; 1 eval = 1 unit |
| Methods | Random / Genetic / Active-learning (same trichotomy) | Direct comparability with Slice 1 |
| Metrics | Convergence curve, success rate, evals-to-target | Reuse Slice 1's metric functions |

## 4. Architecture

New code lives in `atomica/alloy.py` (one focused module; split only if it grows unwieldy). It
reuses Slice 1's `benchmark` logging shape, `plot`, and metric functions.

### Representation

A configuration is a length-12 integer/boolean **site labeling** (e.g. a sorted tuple of the 6 site
indices that hold Au; the rest hold Cu). Fixed lattice geometry is built once.

### Interfaces (the new seams)

- `build_lattice() -> ase.Atoms` — the fixed 12-site FCC supercell (species filled per config).
- `evaluate(config) -> float` — rigid MACE-MP-0 single-point energy of `config`. One call = one
  budget unit. Wraps a module-level cached MACE calculator.
- `sro_descriptor(config) -> np.ndarray` — a fixed-length short-range-order vector: counts of
  first-nearest-neighbour pairs by type (Au–Au, Au–Cu, Cu–Cu), normalized. Uses a neighbour list of
  the fixed lattice computed once (FCC first-NN distance ≈ a/√2 ≈ 2.72 Å; cutoff ≈ 2.9 Å).
- Three search functions sharing one signature:
  `search(evaluate, n_sites, n_au, budget, seed) -> (history, best_config)`
  where `history` is a list of `(n_evals_used, best_energy_so_far)` — **the same shape Slice 1's
  benchmark/plot already consume**.
- `brute_force_min(evaluate, n_sites, n_au) -> (min_energy, best_config, all_energies)` — enumerate
  and cache the ground truth.

### Search operators (all composition-preserving: always exactly `n_au` Au)

- **Random:** `rng.choice(n_sites, n_au, replace=False)` as the Au sites; evaluate; keep the best.
- **Genetic:** a population of labelings; selection by energy; **composition-preserving crossover**
  (child inherits the Au sites shared by both parents, then fills the remaining Au slots by random
  choice among the sites that are Au in exactly one parent, until it has exactly `n_au`);
  **swap mutation** (exchange one Au site for one Cu site). Keep the fittest each generation.
- **Active-learning:** maintain a dataset `(sro_descriptor(config) -> energy)`; fit a
  `RandomForestRegressor`; each round generate a candidate pool (random configs + swap-mutations of
  the current best), predict energy and uncertainty (std across trees), score by lower-confidence-
  bound `mean − k·std`, evaluate the argmin (spending one budget unit), add it, retrain. Surrogate
  predictions over the pool are free; only `evaluate` calls are charged — the same fairness invariant
  as Slice 1.

### Reuse of the Slice 1 harness

- **`benchmark`:** Slice 1's `run_benchmark` calls `fn(n, budget, seed, relax)`. Generalize it (or
  add a thin alloy runner) so a search receives a generic **`evaluate` callable** instead of `relax`,
  and the fixed problem parameters (`n_sites`, `n_au`) pass through. The per-run JSON schema is
  unchanged (`method, n, seed, budget, history, best_energy, best_config`), so `plot` and the metric
  functions keep working. `best_positions` becomes `best_config` (a list of site indices).
- **`plot` / metrics:** `success_rate`, `evals_to_target`, and `write_metrics` already take an
  explicit `target`; feed them the brute-forced minimum energy. `make_figures` / `write_metrics` take
  a `known_minima`-style override (the brute-forced value) instead of the hard-coded LJ dict.
- **Slice 1 must keep working** — the LJ path is untouched; generalization is additive.

### Data flow

```
brute_force_min (once, cached JSON) ─┐  target
run_alloy_benchmark                  │
  └─ for method × seed:              ▼
       alloy search ── evaluate ─▶ MACE-MP-0 single-point (cached calc)
          │  (active only) └─▶ sro_descriptor
          └─▶ history → results/<method>_alloy_seed<s>.json
plot/metrics: histories + brute-forced target → convergence curve + success/evals metrics
```

## 5. Testing & validation

- **Composition invariant (key):** every operator (`random`, `mutate`, `crossover`) yields a config
  with exactly `n_au` Au sites — assert across many random draws.
- **Evaluate sanity:** `evaluate` on a valid config returns a finite, negative energy in a physical
  range (order −4 eV/atom); two configs that are relabelings of each other by a lattice symmetry give
  equal energy.
- **SRO descriptor:** fixed length; pair counts sum to the total number of first-NN pairs; a config
  and a symmetry-equivalent relabeling give the same descriptor.
- **Search plumbing:** each method returns a non-increasing `history` of length `budget` and a valid
  best config; on a tiny budget it runs end-to-end.
- **Budget fairness:** wrap `evaluate` in a counter; assert the active-learning method calls it
  exactly `budget` times (surrogate predictions free) — the P2 analogue of Slice 1's call-count test.
- **Ground-truth consistency:** `brute_force_min` over the 924 configs returns a minimum ≤ every
  sampled energy; cache is deterministic.

## 6. Deliverable / research question

> Under an equal budget of MACE single-point evaluations, does the active-learning method find the
> true (brute-forced) ground-state Cu-Au ordering faster than Random and Genetic?

Plus the bonus physics observation: does MACE's ground state match the known L1₀-type ordering? As in
Slice 1, a negative result (active-learning does not win) is a valid, honestly-reported finding.

## 7. Dependencies & scope

- Add `mace-torch` to `requirements.txt` (MACE-MP-0 weights auto-download and cache on first use).
- **In scope:** the alloy module, brute-forced ground truth, the three discrete search methods, the
  benchmark generalization, reused plots/metrics, tests.
- **Out of scope (later phases):** per-config relaxation studies, other alloys/compositions/sizes,
  larger cells beyond brute-force range, any LLM component (P3+).

## 8. Open items to resolve during implementation

- Confirm the exact `mace_mp` call and that CPU/float64 give deterministic single-point energies
  across runs (needed for reproducible ground truth).
- Confirm the FCC first-NN cutoff for the neighbour list on the chosen supercell (PBC).
- Decide the active-learning knobs (candidate-pool size, `k_acq`, `n_init`) — tune only after the
  harness is validated on the brute-forced target.
- Choose `tol` for "found the global minimum" (energy tolerance vs exact config match).
