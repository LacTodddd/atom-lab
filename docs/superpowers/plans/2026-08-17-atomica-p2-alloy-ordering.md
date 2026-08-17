# ATOMICA P2 — Alloy-Ordering Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark Random vs Genetic vs Active-learning search for the lowest-energy Cu-Au ordering on a fixed 12-site FCC lattice, evaluated with the real MACE-MP-0 potential, against a brute-forced ground truth.

**Architecture:** A configuration is a `tuple(sorted(au_site_indices))` of length 6 on a fixed 12-site FCC supercell. `atomica/alloy.py` holds the fixed lattice, a cached MACE single-point `evaluate`, a short-range-order descriptor, and the brute-force ground truth. `atomica/alloy_search.py` holds composition-preserving operators and the three discrete search methods (shared signature). Slice 1's `benchmark`/`plot`/metrics are reused via a small additive generalization; the LJ path is untouched.

**Tech Stack:** Python 3.13, `mace-torch` (MACE-MP-0), `ase` (lattice + neighbour list), `numpy`, `scikit-learn` (RandomForest surrogate). CPU, float64.

**Spec:** `docs/superpowers/specs/2026-08-17-atomica-p2-alloy-ordering-design.md`

## Global Constraints

- **Potential:** MACE-MP-0 `model="small"`, `device="cpu"`, `default_dtype="float64"`, loaded once at module level and cached.
- **Problem is fixed:** 12-site FCC supercell (`bulk('Cu','fcc',a=3.85,cubic=True).repeat((1,1,3))`), composition exactly 6 Au + 6 Cu.
- **Config representation:** `tuple(sorted(au_indices))`, length `n_au=6`, values in `range(n_sites=12)`.
- **Budget = number of `evaluate` calls.** One `evaluate(config)` = one budget unit. `len(history) == budget`. For active-learning, ONLY `evaluate` calls count — surrogate `.predict` over the candidate pool is free.
- **Search signature (all three):** `search(evaluate, n_sites, n_au, budget, seed) -> (history, best_config)` where `history` is a list of `(n_evals_used, best_energy_so_far)` and `best_config` is a `tuple`. Best-energy history is non-increasing.
- **Composition invariant:** every operator (`random_config`, `mutate_swap`, `crossover`) returns a config with exactly `n_au` Au sites.
- **Reproducibility:** seed via `numpy.random.default_rng(seed)`.
- **Slice 1 (LJ) must keep passing** — all changes to `benchmark.py`/`plot.py` are additive.
- **Test speed:** search/benchmark tests use a fast deterministic *fake* `evaluate` (no MACE). Only Task 1's test and the Task 8 deliverable call the real MACE potential.

---

### Task 1: `alloy.py` — fixed lattice + MACE `evaluate`

**Files:**
- Create: `atomica/alloy.py`
- Test: `tests/test_alloy.py`
- Modify: `requirements.txt` (add `mace-torch`)

**Interfaces:**
- Produces:
  - `build_lattice(n_sites=12) -> ase.Atoms` (all-Cu 12-site FCC supercell)
  - `config_symbols(config, n_sites=12) -> list[str]` (`'Au'` at config indices, else `'Cu'`)
  - `evaluate(config, n_sites=12) -> float` (rigid MACE single-point energy; one budget unit)

- [ ] **Step 1: Add dependency**

Append `mace-torch` to `requirements.txt`, then:

```bash
python3 -m pip install mace-torch
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_alloy.py
import numpy as np
from atomica.alloy import build_lattice, config_symbols, evaluate

def test_lattice_has_12_sites():
    at = build_lattice()
    assert len(at) == 12

def test_config_symbols_composition():
    syms = config_symbols((0, 1, 2, 3, 4, 5))
    assert syms.count("Au") == 6 and syms.count("Cu") == 6

def test_evaluate_is_sane_and_symmetry_consistent():
    # A valid 6-Au config gives a finite, physical energy (~ -4 eV/atom for Cu-Au).
    e = evaluate((0, 1, 2, 3, 4, 5))
    assert np.isfinite(e)
    assert -6.0 * 12 < e < 0.0
    # Translating the whole labeling by the supercell period (sites 0..3 -> 4..7) is a
    # symmetry-equivalent config and must give (near-)equal energy.
    e_shift = evaluate((4, 5, 6, 7, 8, 9))
    assert abs(e - e_shift) < 1e-3
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_alloy.py -v`
Expected: FAIL with `ModuleNotFoundError: atomica.alloy`.

- [ ] **Step 4: Write minimal implementation**

```python
# atomica/alloy.py
import numpy as np
from ase.build import bulk

N_SITES = 12
N_AU = 6
A_LATTICE = 3.85  # Vegard mean of Cu (3.615) and Au (4.078)

def build_lattice(n_sites=N_SITES):
    at = bulk("Cu", "fcc", a=A_LATTICE, cubic=True).repeat((1, 1, 3))
    assert len(at) == n_sites
    return at

def config_symbols(config, n_sites=N_SITES):
    symbols = ["Cu"] * n_sites
    for i in config:
        symbols[i] = "Au"
    return symbols

_CALC = None
def _calc():
    global _CALC
    if _CALC is None:
        from mace.calculators import mace_mp
        _CALC = mace_mp(model="small", dispersion=False,
                        default_dtype="float64", device="cpu")
    return _CALC

def evaluate(config, n_sites=N_SITES):
    at = build_lattice(n_sites)
    at.symbols = config_symbols(config, n_sites)
    at.calc = _calc()
    return float(at.get_potential_energy())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_alloy.py -v`
Expected: PASS (first run downloads/caches the MACE model; ~6 s load).

- [ ] **Step 6: Commit**

```bash
git add atomica/alloy.py tests/test_alloy.py requirements.txt
git commit -m "feat: fixed Cu-Au FCC lattice and MACE single-point evaluate"
```

---

### Task 2: `alloy.py` — neighbour list + SRO descriptor

**Files:**
- Modify: `atomica/alloy.py`
- Test: `tests/test_alloy.py` (add cases)

**Interfaces:**
- Produces: `sro_descriptor(config, n_sites=12) -> np.ndarray` — length-3 normalized first-NN pair counts `[Au-Au, Au-Cu, Cu-Cu]`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_alloy.py
from atomica.alloy import sro_descriptor

def test_sro_fixed_length_and_normalized():
    d = sro_descriptor((0, 1, 2, 3, 4, 5))
    assert d.shape == (3,)
    assert abs(d.sum() - 1.0) < 1e-9

def test_sro_symmetry_equivalent_configs_match():
    # Period shift is a lattice symmetry -> identical SRO.
    a = sro_descriptor((0, 1, 2, 3, 4, 5))
    b = sro_descriptor((4, 5, 6, 7, 8, 9))
    assert np.allclose(a, b)

def test_sro_all_au_pairs_only_auau():
    # If every neighbour bond is Au-Au (all 12 sites Au — not composition-valid, but a pure
    # descriptor check), Au-Cu and Cu-Cu bins are zero.
    d = sro_descriptor(tuple(range(12)))
    assert d[1] == 0.0 and d[2] == 0.0 and abs(d[0] - 1.0) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_alloy.py -k sro -v`
Expected: FAIL with ImportError for `sro_descriptor`.

- [ ] **Step 3: Write minimal implementation** (append to `atomica/alloy.py`)

```python
_NBR = None  # directed (i, j) first-nearest-neighbour index arrays for the fixed lattice
def _neighbours(cutoff=2.9):
    # FCC first-NN distance is a/sqrt(2) ~= 2.72 A; 2.9 A captures first NN only (second is 3.85).
    global _NBR
    if _NBR is None:
        from ase.neighborlist import neighbor_list
        i, j = neighbor_list("ij", build_lattice(), cutoff)
        _NBR = (np.asarray(i), np.asarray(j))
    return _NBR

def sro_descriptor(config, n_sites=N_SITES):
    is_au = np.zeros(n_sites, dtype=bool)
    is_au[list(config)] = True
    i, j = _neighbours()
    ai, aj = is_au[i], is_au[j]
    # directed bonds: each undirected bond counted twice -> divide by 2
    counts = np.array([np.sum(ai & aj), np.sum(ai ^ aj), np.sum(~ai & ~aj)], dtype=float) / 2.0
    total = counts.sum()
    return counts / total if total > 0 else counts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_alloy.py -k sro -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/alloy.py tests/test_alloy.py
git commit -m "feat: short-range-order descriptor for alloy configs"
```

---

### Task 3: `alloy.py` — brute-force ground truth

**Files:**
- Modify: `atomica/alloy.py`
- Test: `tests/test_alloy.py` (add case)

**Interfaces:**
- Produces: `brute_force_min(evaluate_fn, n_sites=12, n_au=6, cache_path=None) -> tuple[float, tuple, int]` returning `(min_energy, best_config, n_evaluated)`; writes `cache_path` JSON if given and reuses it if present.

- [ ] **Step 1: Write the failing test** (uses a fast fake evaluate + small space, not MACE)

```python
# add to tests/test_alloy.py
from atomica.alloy import brute_force_min

def test_brute_force_finds_min_on_toy():
    # Fake energy: lower when sites {0,1} are chosen. Space = C(4,2) = 6 configs.
    def fake(config, n_sites=4):
        return float(sum(config))  # minimized by (0,1)
    e, cfg, n = brute_force_min(fake, n_sites=4, n_au=2)
    assert n == 6
    assert cfg == (0, 1)
    assert e == 1.0

def test_brute_force_caches(tmp_path):
    calls = {"n": 0}
    def fake(config, n_sites=4):
        calls["n"] += 1
        return float(sum(config))
    p = tmp_path / "gt.json"
    brute_force_min(fake, n_sites=4, n_au=2, cache_path=p)
    first = calls["n"]
    brute_force_min(fake, n_sites=4, n_au=2, cache_path=p)  # served from cache
    assert calls["n"] == first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_alloy.py -k brute -v`
Expected: FAIL with ImportError for `brute_force_min`.

- [ ] **Step 3: Write minimal implementation** (append to `atomica/alloy.py`)

```python
import json
from itertools import combinations
from pathlib import Path

def brute_force_min(evaluate_fn, n_sites=N_SITES, n_au=N_AU, cache_path=None):
    if cache_path is not None and Path(cache_path).exists():
        d = json.loads(Path(cache_path).read_text())
        return d["min_energy"], tuple(d["best_config"]), d["n_evaluated"]
    best_e, best_c, n = float("inf"), None, 0
    for config in combinations(range(n_sites), n_au):
        e = evaluate_fn(config)
        n += 1
        if e < best_e:
            best_e, best_c = e, config
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(
            {"min_energy": best_e, "best_config": list(best_c), "n_evaluated": n}))
    return best_e, best_c, n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_alloy.py -k brute -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/alloy.py tests/test_alloy.py
git commit -m "feat: brute-force alloy ground truth with caching"
```

---

### Task 4: `alloy_search.py` — operators + `random_search`

**Files:**
- Create: `atomica/alloy_search.py`
- Test: `tests/test_alloy_search.py`

**Interfaces:**
- Produces:
  - `random_config(n_sites, n_au, rng) -> tuple`
  - `mutate_swap(config, n_sites, rng) -> tuple`
  - `random_search(evaluate, n_sites, n_au, budget, seed) -> (history, best_config)`

- [ ] **Step 1: Write the failing tests** (fast fake evaluate)

```python
# tests/test_alloy_search.py
import numpy as np
from atomica.alloy_search import random_config, mutate_swap, random_search

def _fake(config, n_sites=12):
    return float(sum(config))  # deterministic, minimized by choosing the lowest indices

def test_random_config_composition():
    rng = np.random.default_rng(0)
    for _ in range(50):
        c = random_config(12, 6, rng)
        assert len(set(c)) == 6 and all(0 <= i < 12 for i in c)

def test_mutate_preserves_composition():
    rng = np.random.default_rng(0)
    c = (0, 1, 2, 3, 4, 5)
    for _ in range(50):
        c = mutate_swap(c, 12, rng)
        assert len(set(c)) == 6

def test_random_search_history_valid():
    hist, best = random_search(_fake, 12, 6, budget=20, seed=0)
    assert len(hist) == 20
    assert [h[0] for h in hist] == list(range(1, 21))
    energies = [h[1] for h in hist]
    assert all(energies[i] >= energies[i + 1] - 1e-9 for i in range(len(energies) - 1))
    assert len(set(best)) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_alloy_search.py -v`
Expected: FAIL with `ModuleNotFoundError: atomica.alloy_search`.

- [ ] **Step 3: Write minimal implementation**

```python
# atomica/alloy_search.py
import numpy as np

def random_config(n_sites, n_au, rng):
    return tuple(sorted(int(x) for x in rng.choice(n_sites, n_au, replace=False)))

def mutate_swap(config, n_sites, rng):
    au = set(config)
    cu = [s for s in range(n_sites) if s not in au]
    out = int(rng.choice(list(au)))
    inn = int(rng.choice(cu))
    au.discard(out)
    au.add(inn)
    return tuple(sorted(au))

def random_search(evaluate, n_sites, n_au, budget, seed):
    rng = np.random.default_rng(seed)
    best_e, best_c = np.inf, None
    history = []
    for i in range(budget):
        c = random_config(n_sites, n_au, rng)
        e = evaluate(c)
        if e < best_e:
            best_e, best_c = e, c
        history.append((i + 1, best_e))
    return history, best_c
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_alloy_search.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/alloy_search.py tests/test_alloy_search.py
git commit -m "feat: alloy random search + composition-preserving operators"
```

---

### Task 5: `alloy_search.py` — `genetic_search`

**Files:**
- Modify: `atomica/alloy_search.py`
- Test: `tests/test_alloy_search.py` (add cases)

**Interfaces:**
- Consumes: `random_config`, `mutate_swap`.
- Produces:
  - `crossover(p1, p2, n_sites, n_au, rng) -> tuple`
  - `genetic_search(evaluate, n_sites, n_au, budget, seed, pop_size=10) -> (history, best_config)`

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_alloy_search.py
from atomica.alloy_search import crossover, genetic_search

def test_crossover_preserves_composition():
    rng = np.random.default_rng(0)
    p1, p2 = (0, 1, 2, 3, 4, 5), (2, 3, 6, 7, 8, 9)
    for _ in range(50):
        child = crossover(p1, p2, 12, 6, rng)
        assert len(set(child)) == 6

def test_genetic_search_history_valid():
    hist, best = genetic_search(_fake, 12, 6, budget=25, seed=1)
    assert len(hist) == 25
    energies = [h[1] for h in hist]
    assert all(energies[i] >= energies[i + 1] - 1e-9 for i in range(len(energies) - 1))
    assert len(set(best)) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_alloy_search.py -k "crossover or genetic" -v`
Expected: FAIL with ImportError for `crossover`.

- [ ] **Step 3: Write minimal implementation** (append to `atomica/alloy_search.py`)

```python
def crossover(p1, p2, n_sites, n_au, rng):
    s1, s2 = set(p1), set(p2)
    shared = s1 & s2                      # Au in both parents -> always kept
    only_one = list(s1 ^ s2)             # Au in exactly one parent
    need = n_au - len(shared)
    chosen = rng.choice(only_one, need, replace=False) if need > 0 else []
    return tuple(sorted(shared | {int(x) for x in chosen}))

def genetic_search(evaluate, n_sites, n_au, budget, seed, pop_size=10):
    rng = np.random.default_rng(seed)
    used = 0
    history = []
    best_e, best_c = np.inf, None
    pop = []  # list of (energy, config)

    def record(e, c):
        nonlocal best_e, best_c, used
        used += 1
        if e < best_e:
            best_e, best_c = e, c
        history.append((used, best_e))

    for _ in range(min(pop_size, budget)):
        c = random_config(n_sites, n_au, rng)
        e = evaluate(c)
        record(e, c)
        pop.append((e, c))

    while used < budget:
        pop.sort(key=lambda t: t[0])
        parents = pop[: max(2, pop_size // 2)]
        pa = parents[int(rng.integers(len(parents)))][1]
        pb = parents[int(rng.integers(len(parents)))][1]
        child = mutate_swap(crossover(pa, pb, n_sites, n_au, rng), n_sites, rng)
        e = evaluate(child)
        record(e, child)
        pop.append((e, child))
        pop.sort(key=lambda t: t[0])
        pop = pop[:pop_size]

    return history, best_c
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_alloy_search.py -k "crossover or genetic" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/alloy_search.py tests/test_alloy_search.py
git commit -m "feat: alloy genetic search with composition-preserving crossover"
```

---

### Task 6: `alloy_search.py` — `active_learning_search`

**Files:**
- Modify: `atomica/alloy_search.py`
- Test: `tests/test_alloy_search.py` (add cases)

**Interfaces:**
- Consumes: `random_config`, `mutate_swap`, `atomica.alloy.sro_descriptor`, `sklearn.ensemble.RandomForestRegressor`.
- Produces: `active_learning_search(evaluate, n_sites, n_au, budget, seed, n_init=10, pool=80, k_acq=1.0) -> (history, best_config)`

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_alloy_search.py
from atomica.alloy_search import active_learning_search

def test_active_learning_history_valid():
    hist, best = active_learning_search(_fake, 12, 6, budget=20, seed=2)
    assert len(hist) == 20
    energies = [h[1] for h in hist]
    assert all(energies[i] >= energies[i + 1] - 1e-9 for i in range(len(energies) - 1))
    assert len(set(best)) == 6

def test_active_learning_calls_evaluate_exactly_budget():
    calls = {"n": 0}
    def counting(config, n_sites=12):
        calls["n"] += 1
        return _fake(config)
    active_learning_search(counting, 12, 6, budget=18, seed=0)
    assert calls["n"] == 18  # surrogate predictions cost no budget
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_alloy_search.py -k active -v`
Expected: FAIL with ImportError for `active_learning_search`.

- [ ] **Step 3: Write minimal implementation** (append to `atomica/alloy_search.py`)

```python
from sklearn.ensemble import RandomForestRegressor
from atomica.alloy import sro_descriptor

def active_learning_search(evaluate, n_sites, n_au, budget, seed,
                           n_init=10, pool=80, k_acq=1.0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    best_e, best_c = np.inf, None
    history = []
    used = 0

    def record(c, e):
        nonlocal best_e, best_c, used
        used += 1
        X.append(sro_descriptor(c, n_sites))
        y.append(e)
        if e < best_e:
            best_e, best_c = e, c
        history.append((used, best_e))

    for _ in range(min(n_init, budget)):
        c = random_config(n_sites, n_au, rng)
        record(c, evaluate(c))

    while used < budget:
        model = RandomForestRegressor(n_estimators=100, random_state=seed)
        model.fit(np.array(X), np.array(y))
        cands = [random_config(n_sites, n_au, rng) for _ in range(pool // 2)]
        cands += [mutate_swap(best_c, n_sites, rng) for _ in range(pool - pool // 2)]
        D = np.array([sro_descriptor(c, n_sites) for c in cands])
        preds = np.stack([est.predict(D) for est in model.estimators_])  # (trees, pool)
        acq = preds.mean(0) - k_acq * preds.std(0)   # lower-confidence-bound (minimizing)
        pick = cands[int(np.argmin(acq))]
        record(pick, evaluate(pick))

    return history, best_c
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_alloy_search.py -k active -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/alloy_search.py tests/test_alloy_search.py
git commit -m "feat: alloy active-learning search (RandomForest + SRO + LCB)"
```

---

### Task 7: Generalize `benchmark.py` + `plot.py` for the alloy problem

**Files:**
- Modify: `atomica/benchmark.py`
- Modify: `atomica/plot.py`
- Test: `tests/test_alloy_benchmark.py`

**Interfaces:**
- Produces (benchmark.py): `run_alloy_benchmark(methods, seeds, budget, evaluate, n_sites, n_au, out_dir="results") -> list[str]`. Writes one JSON per (method, seed) with keys `method, n, seed, budget, history ([step,energy]), best_energy, best_config`.
- Modifies (plot.py): `make_figures` and `write_metrics` accept an optional `known_minima=None` argument that, when given, overrides the module-level `KNOWN_MINIMA` (so the alloy target can be supplied). Default behaviour (LJ) is unchanged.

- [ ] **Step 1: Write the failing test** (fast fake evaluate)

```python
# tests/test_alloy_benchmark.py
import json
from atomica.benchmark import run_alloy_benchmark
from atomica.alloy_search import random_search

def _fake(config, n_sites=12):
    return float(sum(config))

def test_run_alloy_benchmark_writes_json(tmp_path):
    paths = run_alloy_benchmark({"random": random_search}, seeds=[0, 1], budget=6,
                                evaluate=_fake, n_sites=12, n_au=6, out_dir=tmp_path)
    assert len(paths) == 2
    d = json.loads(open(paths[0]).read())
    assert d["method"] == "random" and d["budget"] == 6 and d["n"] == 12
    assert len(d["history"]) == 6
    assert len(d["best_config"]) == 6
```

Also add a plot override test:

```python
# tests/test_plot.py  (add)
from atomica.plot import success_rate  # existing
from atomica.plot import make_figures  # ensure importable with new kwarg
def test_make_figures_accepts_known_minima_override(tmp_path):
    import json
    (tmp_path / "random_N12_seed0.json").write_text(json.dumps(
        {"method": "random", "n": 12, "seed": 0, "budget": 2,
         "history": [[1, -1.0], [2, -2.0]], "best_energy": -2.0, "best_config": [0,1,2,3,4,5]}))
    out = make_figures(results_dir=tmp_path, out_dir=tmp_path, known_minima={12: -2.5})
    assert out  # a PNG path was produced
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_alloy_benchmark.py tests/test_plot.py -k "alloy or override" -v`
Expected: FAIL (`run_alloy_benchmark` missing; `make_figures` has no `known_minima` kwarg).

- [ ] **Step 3: Write minimal implementations**

Append to `atomica/benchmark.py`:

```python
def run_alloy_benchmark(methods, seeds, budget, evaluate, n_sites, n_au, out_dir="results"):
    from pathlib import Path
    import json
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn in methods.items():
        for seed in seeds:
            history, best = fn(evaluate, n_sites, n_au, budget, seed)
            path = out / f"{name}_N{n_sites}_seed{seed}.json"
            path.write_text(json.dumps({
                "method": name, "n": int(n_sites), "seed": int(seed), "budget": int(budget),
                "history": [[int(s), float(e)] for s, e in history],
                "best_energy": float(history[-1][1]),
                "best_config": [int(i) for i in best],
            }))
            written.append(str(path))
    return written
```

In `atomica/plot.py`, change the signatures of `make_figures` and `write_metrics` to accept `known_minima=None` and use it when provided:

```python
def make_figures(results_dir="results", out_dir="results", known_minima=None):
    km = KNOWN_MINIMA if known_minima is None else known_minima
    # ... existing body, but replace every use of KNOWN_MINIMA with km ...
```

```python
def write_metrics(results_dir="results", out_dir="results", tol=0.01, known_minima=None):
    km = KNOWN_MINIMA if known_minima is None else known_minima
    # ... existing body, but replace every use of KNOWN_MINIMA with km ...
```

(Leave the module-level `KNOWN_MINIMA` and all existing call sites/tests intact — the new kwarg defaults to it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_alloy_benchmark.py tests/test_plot.py -v`
Expected: PASS. Then run the whole suite to confirm the LJ path still works: `python3 -m pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add atomica/benchmark.py atomica/plot.py tests/test_alloy_benchmark.py tests/test_plot.py
git commit -m "feat: alloy benchmark runner + known_minima override for plots/metrics"
```

---

### Task 8: `run_alloy.py` — ground truth + full real-MACE deliverable

**Files:**
- Create: `atomica/run_alloy.py`
- Test: `tests/test_run_alloy.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `main(argv=None) -> None` and a `python -m atomica.run_alloy` entry point that: computes/caches the brute-force ground truth, runs the 3-method × N-seed × budget benchmark, and writes convergence figures + metrics against the ground-truth target.

- [ ] **Step 1: Write the failing test** (fast: monkeypatch `evaluate` with a fake so the CLI wiring is tested without MACE)

```python
# tests/test_run_alloy.py
from atomica import run_alloy

def test_cli_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(run_alloy, "evaluate", lambda config, n_sites=12: float(sum(config)))
    run_alloy.main(["--budget", "6", "--seeds", "2", "--methods", "random",
                    "--out", str(tmp_path)])
    assert (tmp_path / "alloy_ground_truth.json").exists()
    assert (tmp_path / "random_N12_seed0.json").exists()
    assert list(tmp_path.glob("convergence_N12.png"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_run_alloy.py -v`
Expected: FAIL with `ModuleNotFoundError: atomica.run_alloy`.

- [ ] **Step 3: Write minimal implementation**

```python
# atomica/run_alloy.py
import argparse
from pathlib import Path
from atomica.alloy import evaluate, brute_force_min, N_SITES, N_AU
from atomica.alloy_search import random_search, genetic_search, active_learning_search
from atomica.benchmark import run_alloy_benchmark
from atomica.plot import make_figures, write_metrics

METHODS = {"random": random_search, "genetic": genetic_search, "active": active_learning_search}

def main(argv=None):
    p = argparse.ArgumentParser(description="ATOMICA P2 Cu-Au alloy-ordering benchmark")
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)

    gt_path = Path(a.out) / "alloy_ground_truth.json"
    min_e, best_cfg, _ = brute_force_min(evaluate, N_SITES, N_AU, cache_path=gt_path)

    methods = {name: METHODS[name] for name in a.methods}
    run_alloy_benchmark(methods, list(range(a.seeds)), a.budget,
                        evaluate, N_SITES, N_AU, out_dir=a.out)
    known = {N_SITES: min_e}
    make_figures(results_dir=a.out, out_dir=a.out, known_minima=known)
    write_metrics(results_dir=a.out, out_dir=a.out, known_minima=known)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_run_alloy.py -v`
Expected: PASS.

- [ ] **Step 5: Full real-MACE deliverable run** (heavy — brute force 924 evals (~5 min) + 3×5×100 search evals; expect ~15-25 min)

```bash
python3 -m atomica.run_alloy --budget 100 --seeds 5 --out results
```

Expected: `results/alloy_ground_truth.json`, `results/convergence_N12.png`, `results/metrics_N12.json`. Read off: under equal budget, did `active` find the brute-forced ground state faster than `random`/`genetic`? Also inspect whether the ground-state config corresponds to an L1₀-type layering (bonus physics note).

- [ ] **Step 6: Update `README.md`** — add a "P2 — Cu-Au alloy ordering (real MACE potential)" section: what it does, how to run (`python3 -m atomica.run_alloy`), the result (per-method success/evals vs the brute-forced ground truth), and a pointer to the P2 design spec. Report the negative-or-positive result honestly.

- [ ] **Step 7: Run the full test suite**

Run: `python3 -m pytest -q`
Expected: all pass (Slice 1 LJ tests + P2 alloy tests).

- [ ] **Step 8: Commit**

```bash
git add atomica/run_alloy.py tests/test_run_alloy.py README.md results/convergence_N12.png results/metrics_N12.json
git commit -m "feat: P2 alloy-ordering CLI, ground truth, and benchmark results"
```

---

## Self-Review

**1. Spec coverage:**
- MACE-MP-0 evaluate on fixed Cu-Au FCC lattice (spec §2/§4) → Task 1. ✅
- SRO descriptor (spec §4) → Task 2. ✅
- Brute-force ground truth + cache (spec §2) → Task 3, run in Task 8. ✅
- Random/Genetic/Active-learning, composition-preserving, shared `evaluate`+budget (spec §4) → Tasks 4-6. ✅
- Budget = evaluate calls; surrogate free; same-seed reproducible (spec §2/§4) → per-method history + Task 6 call-count test. ✅
- Reuse benchmark/plot/metrics with target override, LJ path intact (spec §4) → Task 7. ✅
- Brute-forced target as the success/evals reference (spec §5/§6) → Task 8 (`known_minima={12: min_e}`). ✅
- Add `mace-torch` dep (spec §7) → Task 1. ✅
- Tests: composition invariant, SRO invariance, evaluate sanity, call-count fairness, ground-truth consistency (spec §5) → Tasks 1-6. ✅

**2. Placeholder scan:** No "TBD"/"handle edge cases"/placeholder code. Every code step is the actual code to write. Active-learning knobs (`pool`, `k_acq`, `n_init`) have concrete defaults.

**3. Type consistency:** `evaluate(config)->float` used identically in Tasks 1,3,4,5,6,7,8. All three `*_search` share `(evaluate, n_sites, n_au, budget, seed)->(history, best_config)`, `history` = list of `(step, energy)`. `sro_descriptor(config, n_sites)->(3,)` in Tasks 2,6. `config` is `tuple(sorted(...))` throughout. `run_alloy_benchmark` JSON keys (`method,n,seed,budget,history,best_energy,best_config`) match what `plot`/metrics consume (`n`, `history` `[step,energy]`). `make_figures`/`write_metrics` `known_minima` kwarg added in Task 7, used in Task 8.
