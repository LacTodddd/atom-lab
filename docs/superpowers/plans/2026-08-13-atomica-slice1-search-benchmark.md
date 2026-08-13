# ATOMICA Slice 1 — LJ-Cluster Search Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark Random vs Genetic vs Active-learning search on Lennard-Jones cluster global-minimum finding, under an equal budget of local relaxations, and produce convergence/success plots.

**Architecture:** A tiny Python package `atomica/`. Clusters are `(N,3)` numpy arrays. A single `relax()` (ASE LJ calculator + local optimizer) is the shared, budget-counted evaluation used by all three search strategies. A benchmark loop writes per-run JSON; a plot module turns JSON into figures. No LLM.

**Tech Stack:** Python 3.13 (miniconda base), `ase` (LJ calculator + optimizer), `numpy`/`scipy` (geometry, pairwise distances), `scikit-learn` (`RandomForestRegressor` surrogate), `matplotlib` (plots). `torch` already present but unused in Slice 1.

**Spec:** `docs/superpowers/specs/2026-08-13-atomica-slice1-design.md`

## Global Constraints

- **Python 3.13, arm64 macOS, 8 GB RAM.** Keep everything CPU-only; no GPU, no ML potentials in Slice 1.
- **Reduced LJ units: ε=1, σ=1.** The LJ calculator MUST be effectively untruncated (`rc` large, `smooth=False`) so energies match the Cambridge Cluster Database reference minima.
- **Budget is measured in local relaxations.** One `relax()` call = one budget unit. Surrogate predictions are free.
- **Reproducibility:** every run is seeded via `numpy.random.default_rng(seed)`. The first K random clusters a method draws from a given `seed` must be identical across methods (fair shared start).
- **Reference global minima (confirm exact values against the Cambridge Cluster Database during Task 7):** N=13 ≈ −44.326801, N=38 ≈ −173.928427.
- **Search signature (all three strategies):**
  `search(n: int, budget: int, seed: int, relax: Callable) -> tuple[list[tuple[int, float]], np.ndarray]`
  returning `(history, best_positions)` where `history` is a list of `(n_relaxations_used, best_energy_so_far)`.
- **No LLM, no agents, no memory/ directories, no web.** Those are later phases.

---

### Task 0: Project setup & environment

**Files:**
- Create: `atomica/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)
- Create: `.gitignore`
- Create: `requirements.txt`

**Interfaces:**
- Produces: an importable `atomica` package and a working `ase` install.

- [ ] **Step 1: Initialize git and package dirs**

```bash
cd "/Users/lactod/Work/Hobby/I'm Atom"
git init
mkdir -p atomica tests results
touch atomica/__init__.py tests/__init__.py
```

- [ ] **Step 2: Install dependencies**

```bash
python3 -m pip install ase matplotlib
```

Create `requirements.txt`:

```
ase
matplotlib
numpy
scipy
scikit-learn
```

- [ ] **Step 3: Verify ASE LJ calculator imports and runs**

Run:

```bash
python3 -c "from ase import Atoms; from ase.calculators.lj import LennardJones; from ase.optimize import FIRE; print('ase ok')"
```

Expected: prints `ase ok` with no ImportError. If `ase` fails to build on Python 3.13, pin a recent version (`pip install 'ase>=3.23'`) and re-run.

- [ ] **Step 4: Create `.gitignore`**

```
__pycache__/
*.pyc
results/*.json
results/*.png
.venv/
```

- [ ] **Step 5: Commit**

```bash
git add atomica tests requirements.txt .gitignore docs
git commit -m "chore: scaffold atomica package and deps"
```

---

### Task 1: `potential.py` — LJ energy + local relaxation

**Files:**
- Create: `atomica/potential.py`
- Test: `tests/test_potential.py`

**Interfaces:**
- Produces:
  - `make_atoms(positions: np.ndarray) -> ase.Atoms` (H atoms + LJ calculator)
  - `relax(positions: np.ndarray, fmax: float = 1e-3, steps: int = 300) -> tuple[np.ndarray, float]`
    returns `(relaxed_positions, energy)`; **one call = one budget unit**.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_potential.py
import numpy as np
from atomica.potential import relax

def test_dimer_relaxes_to_minus_one():
    # Two atoms far apart relax to the LJ minimum energy of -1.0 (reduced units).
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    relaxed, e = relax(pos)
    assert abs(e - (-1.0)) < 1e-2

def test_lj13_icosahedron_matches_reference():
    # The LJ-13 global minimum is a centered icosahedron ~ -44.3268 (reduced units).
    phi = (1 + 5 ** 0.5) / 2
    verts = []
    for a, b in [(1, phi), (-1, phi), (1, -phi), (-1, -phi)]:
        verts += [(0, a, b), (a, b, 0), (b, 0, a)]
    pos = np.vstack([np.array(verts), [0, 0, 0]])          # 12 vertices + center = 13
    pos = pos * (1.12 / np.sqrt(1 + phi ** 2))             # scale near LJ nearest-neighbor
    relaxed, e = relax(pos)
    REF = -44.326801  # confirm against Cambridge Cluster Database
    assert abs(e - REF) < 0.05, f"got {e}; if wrong, check rc/smooth and REF"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_potential.py -v`
Expected: FAIL with `ModuleNotFoundError: atomica.potential` (or `relax` undefined).

- [ ] **Step 3: Write minimal implementation**

```python
# atomica/potential.py
import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import FIRE

# Untruncated LJ so energies match reference minima (rc large, no smoothing).
_CALC_KW = dict(epsilon=1.0, sigma=1.0, rc=500.0, smooth=False)

def make_atoms(positions: np.ndarray) -> Atoms:
    positions = np.asarray(positions, dtype=float)
    atoms = Atoms(f"H{len(positions)}", positions=positions)
    atoms.calc = LennardJones(**_CALC_KW)
    return atoms

def relax(positions: np.ndarray, fmax: float = 1e-3, steps: int = 300):
    atoms = make_atoms(positions)
    opt = FIRE(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return atoms.get_positions(), float(atoms.get_potential_energy())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_potential.py -v`
Expected: PASS. If `test_lj13...` fails on the energy value, confirm the reference from the Cambridge Cluster Database and that `rc`/`smooth` give an untruncated LJ.

- [ ] **Step 5: Commit**

```bash
git add atomica/potential.py tests/test_potential.py
git commit -m "feat: LJ potential and local relaxation with LJ-13 validation"
```

---

### Task 2: `descriptor.py` — pairwise-distance histogram

**Files:**
- Create: `atomica/descriptor.py`
- Test: `tests/test_descriptor.py`

**Interfaces:**
- Produces: `distance_histogram(positions: np.ndarray, bins: int = 30, r_max: float = 8.0) -> np.ndarray`
  a normalized, fixed-length (`bins`) vector; invariant to permutation, rotation, translation.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_descriptor.py
import numpy as np
from atomica.descriptor import distance_histogram

def _random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q

def test_fixed_length():
    rng = np.random.default_rng(0)
    h = distance_histogram(rng.normal(size=(13, 3)), bins=30)
    assert h.shape == (30,)
    assert abs(h.sum() - 1.0) < 1e-9

def test_invariant_to_rotation_permutation_translation():
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(13, 3))
    R = _random_rotation(rng)
    perm = rng.permutation(13)
    transformed = pos[perm] @ R.T + np.array([5.0, -2.0, 3.0])
    a = distance_histogram(pos)
    b = distance_histogram(transformed)
    assert np.allclose(a, b, atol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_descriptor.py -v`
Expected: FAIL with `ModuleNotFoundError: atomica.descriptor`.

- [ ] **Step 3: Write minimal implementation**

```python
# atomica/descriptor.py
import numpy as np
from scipy.spatial.distance import pdist

def distance_histogram(positions: np.ndarray, bins: int = 30, r_max: float = 8.0) -> np.ndarray:
    d = pdist(np.asarray(positions, dtype=float))          # all pairwise distances
    hist, _ = np.histogram(d, bins=bins, range=(0.0, r_max))
    total = hist.sum()
    if total == 0:
        return np.zeros(bins)
    return hist / total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_descriptor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/descriptor.py tests/test_descriptor.py
git commit -m "feat: rotation/permutation-invariant distance-histogram descriptor"
```

---

### Task 3: `search.py` — helpers + `random_search`

**Files:**
- Create: `atomica/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Produces:
  - `random_cluster(n: int, rng, min_sep: float = 0.8) -> np.ndarray`
  - `random_search(n, budget, seed, relax) -> tuple[list[tuple[int, float]], np.ndarray]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_search.py
import numpy as np
from atomica.potential import relax
from atomica.search import random_cluster, random_search

def test_random_cluster_shape_and_separation():
    rng = np.random.default_rng(0)
    x = random_cluster(6, rng, min_sep=0.8)
    assert x.shape == (6, 3)
    from scipy.spatial.distance import pdist
    assert pdist(x).min() > 0.8

def test_random_search_history_valid():
    hist, best = random_search(2, budget=15, seed=0, relax=relax)
    assert len(hist) == 15
    steps = [h[0] for h in hist]
    energies = [h[1] for h in hist]
    assert steps == list(range(1, 16))                     # 1..budget
    assert all(energies[i] >= energies[i + 1] - 1e-9 for i in range(len(energies) - 1))  # non-increasing
    assert energies[-1] < -0.9                             # N=2 global min is -1.0
    assert best.shape == (2, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError: atomica.search`.

- [ ] **Step 3: Write minimal implementation**

```python
# atomica/search.py
import numpy as np

def random_cluster(n: int, rng, min_sep: float = 0.8) -> np.ndarray:
    R = 0.5 * n ** (1 / 3) + 1.0
    pts = []
    while len(pts) < n:
        p = rng.uniform(-R, R, size=3)
        if np.linalg.norm(p) > R:
            continue
        if any(np.linalg.norm(p - q) < min_sep for q in pts):
            continue
        pts.append(p)
    return np.array(pts)

def random_search(n, budget, seed, relax):
    rng = np.random.default_rng(seed)
    best_e, best_x = np.inf, None
    history = []
    for i in range(budget):
        x, e = relax(random_cluster(n, rng))
        if e < best_e:
            best_e, best_x = e, x
        history.append((i + 1, best_e))
    return history, best_x
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_search.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/search.py tests/test_search.py
git commit -m "feat: random search baseline + cluster init helper"
```

---

### Task 4: `search.py` — `genetic_search`

**Files:**
- Modify: `atomica/search.py`
- Test: `tests/test_search.py` (add cases)

**Interfaces:**
- Consumes: `random_cluster`, `relax`.
- Produces:
  - `cut_and_splice(a: np.ndarray, b: np.ndarray, rng) -> np.ndarray`
  - `mutate(x: np.ndarray, rng, sigma: float = 0.3, frac: float = 0.3) -> np.ndarray`
  - `genetic_search(n, budget, seed, relax) -> tuple[list[tuple[int, float]], np.ndarray]`

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_search.py
from atomica.search import cut_and_splice, mutate, genetic_search

def test_cut_and_splice_preserves_atom_count():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(13, 3))
    b = rng.normal(size=(13, 3))
    child = cut_and_splice(a, b, rng)
    assert child.shape == (13, 3)

def test_genetic_search_history_valid():
    hist, best = genetic_search(2, budget=20, seed=1, relax=relax)
    assert len(hist) == 20
    energies = [h[1] for h in hist]
    assert all(energies[i] >= energies[i + 1] - 1e-9 for i in range(len(energies) - 1))
    assert energies[-1] < -0.9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_search.py -k "cut_and_splice or genetic" -v`
Expected: FAIL with ImportError for `cut_and_splice`.

- [ ] **Step 3: Write minimal implementation** (append to `atomica/search.py`)

```python
def cut_and_splice(a, b, rng):
    n = len(a)
    a = a - a.mean(0)
    b = b - b.mean(0)
    normal = rng.normal(size=3)
    normal /= np.linalg.norm(normal)
    a_sorted = a[np.argsort(a @ normal)]
    b_sorted = b[np.argsort(b @ normal)]
    k = int(rng.integers(1, n))
    return np.vstack([a_sorted[:k], b_sorted[k:]])

def mutate(x, rng, sigma=0.3, frac=0.3):
    x = x.copy()
    m = rng.random(len(x)) < frac
    if m.any():
        x[m] += rng.normal(scale=sigma, size=(int(m.sum()), 3))
    return x

def genetic_search(n, budget, seed, relax, pop_size=10):
    rng = np.random.default_rng(seed)
    used = 0
    history = []
    pop = []  # list of (energy, positions)
    best_e, best_x = np.inf, None

    def record(e, x):
        nonlocal best_e, best_x, used
        used += 1
        if e < best_e:
            best_e, best_x = e, x
        history.append((used, best_e))

    for _ in range(min(pop_size, budget)):
        x, e = relax(random_cluster(n, rng))
        pop.append((e, x))
        record(e, x)

    while used < budget:
        pop.sort(key=lambda t: t[0])
        parents = pop[: max(2, pop_size // 2)]
        (_, pa), (_, pb) = (parents[int(rng.integers(len(parents)))] for _ in range(2))
        child = mutate(cut_and_splice(pa, pb, rng), rng)
        x, e = relax(child)
        record(e, x)
        pop.append((e, x))
        pop.sort(key=lambda t: t[0])
        pop = pop[:pop_size]                                # keep the fittest

    return history, best_x
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_search.py -k "cut_and_splice or genetic" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/search.py tests/test_search.py
git commit -m "feat: genetic search with cut-and-splice crossover"
```

---

### Task 5: `search.py` — `active_learning_search`

**Files:**
- Modify: `atomica/search.py`
- Test: `tests/test_search.py` (add case)

**Interfaces:**
- Consumes: `random_cluster`, `mutate`, `relax`, `descriptor.distance_histogram`, `sklearn.ensemble.RandomForestRegressor`.
- Produces: `active_learning_search(n, budget, seed, relax) -> tuple[list[tuple[int, float]], np.ndarray]`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_search.py
from atomica.search import active_learning_search

def test_active_learning_history_valid():
    hist, best = active_learning_search(2, budget=18, seed=2, relax=relax)
    assert len(hist) == 18
    energies = [h[1] for h in hist]
    assert all(energies[i] >= energies[i + 1] - 1e-9 for i in range(len(energies) - 1))
    assert energies[-1] < -0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_search.py -k active_learning -v`
Expected: FAIL with ImportError for `active_learning_search`.

- [ ] **Step 3: Write minimal implementation** (append to `atomica/search.py`)

```python
from sklearn.ensemble import RandomForestRegressor
from atomica.descriptor import distance_histogram

def active_learning_search(n, budget, seed, relax, n_init=10, pool=100, k_acq=1.0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    best_e, best_x = np.inf, None
    history = []
    used = 0

    def record(x, e):
        nonlocal best_e, best_x, used
        used += 1
        X.append(distance_histogram(x))
        y.append(e)
        if e < best_e:
            best_e, best_x = e, x
        history.append((used, best_e))

    for _ in range(min(n_init, budget)):
        x, e = relax(random_cluster(n, rng))
        record(x, e)

    while used < budget:
        model = RandomForestRegressor(n_estimators=100, random_state=seed)
        model.fit(np.array(X), np.array(y))
        cands = [random_cluster(n, rng) for _ in range(pool // 2)]
        cands += [mutate(best_x, rng) for _ in range(pool - pool // 2)]
        D = np.array([distance_histogram(c) for c in cands])
        preds = np.stack([est.predict(D) for est in model.estimators_])  # (n_trees, pool)
        acq = preds.mean(0) - k_acq * preds.std(0)         # lower-confidence-bound (minimizing energy)
        x, e = relax(cands[int(np.argmin(acq))])
        record(x, e)

    return history, best_x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_search.py -k active_learning -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/search.py tests/test_search.py
git commit -m "feat: active-learning surrogate search (RandomForest + LCB)"
```

---

### Task 6: `benchmark.py` — run all methods and log JSON

**Files:**
- Create: `atomica/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: the three `*_search` functions, `relax`.
- Produces:
  - `METHODS: dict[str, Callable]` mapping name → search function.
  - `run_benchmark(n_values, seeds, budget, methods=METHODS, out_dir="results") -> list[str]`
    writes one JSON per `(method, n, seed)` and returns the written paths. Each JSON:
    `{"method","n","seed","budget","history":[[step,energy],...],"best_energy":float,"best_positions":[[x,y,z],...]}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark.py
import json
from atomica.benchmark import run_benchmark, METHODS

def test_run_benchmark_writes_expected_files(tmp_path):
    paths = run_benchmark(n_values=[2], seeds=[0], budget=6,
                          methods={"random": METHODS["random"]}, out_dir=tmp_path)
    assert len(paths) == 1
    data = json.loads(open(paths[0]).read())
    assert data["method"] == "random"
    assert data["n"] == 2 and data["seed"] == 0 and data["budget"] == 6
    assert len(data["history"]) == 6
    assert data["best_energy"] < -0.9
    assert len(data["best_positions"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError: atomica.benchmark`.

- [ ] **Step 3: Write minimal implementation**

```python
# atomica/benchmark.py
import json
from pathlib import Path
import numpy as np
from atomica.potential import relax
from atomica.search import random_search, genetic_search, active_learning_search

METHODS = {
    "random": random_search,
    "genetic": genetic_search,
    "active": active_learning_search,
}

def run_benchmark(n_values, seeds, budget, methods=METHODS, out_dir="results"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for n in n_values:
        for name, fn in methods.items():
            for seed in seeds:
                history, best = fn(n, budget, seed, relax)
                path = out / f"{name}_N{n}_seed{seed}.json"
                path.write_text(json.dumps({
                    "method": name, "n": int(n), "seed": int(seed), "budget": int(budget),
                    "history": [[int(s), float(e)] for s, e in history],
                    "best_energy": float(history[-1][1]),
                    "best_positions": np.asarray(best).tolist(),
                }))
                written.append(str(path))
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_benchmark.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/benchmark.py tests/test_benchmark.py
git commit -m "feat: benchmark loop with reproducible JSON logging"
```

---

### Task 7: `plot.py` — metrics + figures

**Files:**
- Create: `atomica/plot.py`
- Test: `tests/test_plot.py`

**Interfaces:**
- Consumes: the JSON files written by Task 6.
- Produces:
  - `KNOWN_MINIMA: dict[int, float]` — reference global minima.
  - `success_rate(histories: list[list], target: float, tol: float = 0.01) -> float`
  - `evals_to_target(history: list, target: float, tol: float = 0.01) -> int | None`
  - `make_figures(results_dir="results", out_dir="results") -> list[str]` — writes PNG(s), returns paths.

- [ ] **Step 1: Write the failing tests** (pure-function metrics only; plotting is smoke-tested)

```python
# tests/test_plot.py
from atomica.plot import success_rate, evals_to_target

def test_evals_to_target_hit_and_miss():
    hist = [[1, -1.0], [2, -3.0], [3, -3.0]]
    assert evals_to_target(hist, target=-3.0, tol=0.01) == 2
    assert evals_to_target(hist, target=-5.0, tol=0.01) is None

def test_success_rate_counts_seeds_reaching_target():
    good = [[1, -3.0]]
    bad = [[1, -1.0]]
    assert success_rate([good, bad, good], target=-3.0, tol=0.01) == 2 / 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_plot.py -v`
Expected: FAIL with `ModuleNotFoundError: atomica.plot`.

- [ ] **Step 3: Write minimal implementation**

```python
# atomica/plot.py
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Confirm exact values against the Cambridge Cluster Database.
KNOWN_MINIMA = {13: -44.326801, 38: -173.928427}

def success_rate(histories, target, tol=0.01):
    hits = sum(1 for h in histories if h[-1][1] <= target + tol)
    return hits / len(histories)

def evals_to_target(history, target, tol=0.01):
    for step, energy in history:
        if energy <= target + tol:
            return step
    return None

def _load(results_dir):
    runs = defaultdict(list)  # (n, method) -> list of run dicts
    for p in Path(results_dir).glob("*.json"):
        d = json.loads(p.read_text())
        runs[(d["n"], d["method"])].append(d)
    return runs

def make_figures(results_dir="results", out_dir="results"):
    runs = _load(results_dir)
    ns = sorted({n for (n, _) in runs})
    written = []
    for n in ns:
        plt.figure()
        for (rn, method), group in sorted(runs.items()):
            if rn != n:
                continue
            budget = group[0]["budget"]
            curves = np.array([[e for _, e in d["history"]] for d in group])  # (seeds, budget)
            mean, std = curves.mean(0), curves.std(0)
            x = np.arange(1, budget + 1)
            plt.plot(x, mean, label=method)
            plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        if n in KNOWN_MINIMA:
            plt.axhline(KNOWN_MINIMA[n], ls="--", color="k", label="global min")
        plt.xlabel("relaxations"); plt.ylabel("best energy"); plt.title(f"LJ-{n}"); plt.legend()
        path = str(Path(out_dir) / f"convergence_N{n}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
        written.append(path)
    return written
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_plot.py -v`
Expected: PASS.

- [ ] **Step 5: Confirm reference minima**

Use the `mattpocock-skills:research` skill (or the Cambridge Cluster Database at `http://doye.chem.ox.ac.uk/jon/structures/LJ/tables.150.html`) to confirm `KNOWN_MINIMA`. Correct the values if they differ, and re-run Task 1's `test_lj13_icosahedron_matches_reference`.

- [ ] **Step 6: Commit**

```bash
git add atomica/plot.py tests/test_plot.py
git commit -m "feat: metrics and convergence plots"
```

---

### Task 8: `run.py` — CLI + end-to-end smoke run

**Files:**
- Create: `atomica/run.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: `run_benchmark`, `make_figures`.
- Produces: `main(argv: list[str] | None = None) -> None` and a `python -m atomica.run` entry point.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run.py
from pathlib import Path
from atomica.run import main

def test_cli_smoke(tmp_path):
    out = tmp_path / "results"
    main(["--n", "2", "--budget", "6", "--seeds", "2", "--methods", "random",
          "--out", str(out)])
    assert (out / "random_N2_seed0.json").exists()
    assert (out / "random_N2_seed1.json").exists()
    assert list(out.glob("convergence_N2.png"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError: atomica.run`.

- [ ] **Step 3: Write minimal implementation**

```python
# atomica/run.py
import argparse
from atomica.benchmark import run_benchmark, METHODS
from atomica.plot import make_figures

def main(argv=None):
    p = argparse.ArgumentParser(description="ATOMICA LJ-cluster search benchmark")
    p.add_argument("--n", type=int, nargs="+", default=[13, 38])
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--seeds", type=int, default=5, help="number of seeds (0..seeds-1)")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)
    methods = {name: METHODS[name] for name in a.methods}
    run_benchmark(a.n, list(range(a.seeds)), a.budget, methods=methods, out_dir=a.out)
    make_figures(results_dir=a.out, out_dir=a.out)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_run.py -v`
Expected: PASS.

- [ ] **Step 5: Full validation run (N=13, the real harness proof)**

Run:

```bash
python3 -m atomica.run --n 13 --budget 200 --seeds 5 --out results
```

Expected: `results/convergence_N13.png` shows every method approaching the dashed global-min line; at least the genetic/active curves reach near −44.33. This is the end-to-end proof the whole loop works.

- [ ] **Step 6: Commit**

```bash
git add atomica/run.py tests/test_run.py
git commit -m "feat: CLI entry point and end-to-end benchmark run"
```

---

### Task 9: Full N=38 benchmark + short README

**Files:**
- Create: `README.md`

**Interfaces:** none (produces the actual experiment result + how-to-run docs).

- [ ] **Step 1: Run the full benchmark** (this is the deliverable; may take a while on the 8 GB Mac)

```bash
python3 -m atomica.run --n 13 38 --budget 200 --seeds 5 --out results
```

Expected: `results/convergence_N13.png` and `results/convergence_N38.png`. Read off the answer: under equal budget, does `active` beat `random` and `genetic`? (A negative result on N=38 is valid — see spec §7.)

- [ ] **Step 2: Write `README.md`**

```markdown
# ATOMICA — Slice 1: LJ-cluster search benchmark

Compares Random vs Genetic vs Active-learning search for Lennard-Jones cluster
global minima, under an equal budget of local relaxations.

## Setup
    python3 -m pip install -r requirements.txt

## Run
    python3 -m atomica.run --n 13 38 --budget 200 --seeds 5

Outputs `results/convergence_N{n}.png` and per-run JSON.

## Tests
    python3 -m pytest -q

See `docs/superpowers/specs/2026-08-13-atomica-slice1-design.md` for design and roadmap.
```

- [ ] **Step 3: Run the full test suite**

Run: `python3 -m pytest -q`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add README.md results/*.png
git commit -m "docs: README and Slice 1 benchmark results"
```

---

## Self-Review

**1. Spec coverage:**
- Toy ASE LJ potential behind swappable calc (spec §3/§4) → Task 1 (`_CALC_KW`, `make_atoms`). ✅
- Distance-histogram descriptor (spec §4) → Task 2. ✅
- Random / Genetic / Active-learning, shared `relax` + budget (spec §3/§4, trichotomy) → Tasks 3–5. ✅
- Fairness = relaxations, same seed → same init (spec §4) → each method draws from `default_rng(seed)`; budget counted per `relax` call. ✅
- Reproducible JSON logging (spec §2 bar) → Task 6. ✅
- Metrics: convergence curve, success rate, evals-to-target (spec §6) → Task 7. ✅
- LJ-13 → LJ-38 validation vs known minima (spec §5) → Task 1 (potential) + Task 8/9 (search). ✅
- CLI (plan §17 of vision) → Task 8. ✅
- Non-goals (LLM, real potentials, memory) → absent by construction. ✅

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". The only deferred value (`KNOWN_MINIMA`) is explicitly flagged with a confirmation step (Task 7 Step 5) and guarded by Task 1's test. ✅

**3. Type consistency:** `relax(positions) -> (positions, energy)` used identically in Tasks 1,3,4,5,6. All `*_search` share `(n, budget, seed, relax) -> (history, best)` with `history` = list of `(step, energy)`. `distance_histogram` returns `(bins,)` in Tasks 2 and 5. `METHODS` keys (`random`/`genetic`/`active`) consistent across Tasks 6 and 8. ✅

Note: the design doc mentioned a 3-tuple history `(n, best_E, struct)`; this plan simplifies to `(step, energy)` in `history` plus a separate `best_positions` (smaller JSON, same information). Deliberate, consistent across all tasks.
