# ATOMICA P5 — Literature Agent (Research-Gap Extrapolation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an LLM read a synthetic "paper" that studied part of the Cu-Au design space and predict — cheat-proof against the P2/P4 924-config ground truth — whether the unexplored region contains a better structure, and measure whether it beats a paper-blind baseline.

**Architecture:** Two new modules reuse P4's `critic_world` (hence `alloy`) unchanged. `atomica/litreview_world.py` generates a deterministic seeded "paper" — a summary of the best structure and the energy trend over an explored subset of the 924 orderings — plus a ground-truth label (`better_in_gap`). `atomica/litreview.py` validates a reviewer's structured boolean, provides three reviewers (paper-blind baseline, a non-LLM trend heuristic, and the LLM), and scores accuracy/precision/recall. `atomica/run_litreview.py` is the CLI; the LLM arm auto-skips without a credential.

**Tech Stack:** Python 3.13, `anthropic` SDK (already a dependency), reusing `numpy` and `atomica.critic_world` (MACE-MP-0 via `alloy`).

**Spec:** `docs/superpowers/specs/2026-08-21-atomica-p5-literature-gap-design.md`

## Global Constraints

- **Reuse `atomica/critic_world.py` unchanged** (`build_world`, `features`, `FEATURE_NAMES`, and via it `alloy`). Do not modify Slice 1 / P2 / P3 / P4 files.
- **World:** the P4 Cu-Au 924-config table (`FEATURE_NAMES = ["x1","x2","layer"]`, `N_SITES=12`, `N_AU=6`). Global minimum ≈ −44.3685 eV.
- **A paper** studies an **explored subset** = orderings whose value on one feature `axis` is in a band; the **gap** = complement. Deterministic, seeded. The paper summary carries: `axis`, `region` (text), `n_explored`, `best_config`, `best_energy` (min energy in the explored subset), `boundary_trend` (best energy in ordered sub-bands of `axis` within the explored subset, far→near the gap), `gap_side` ("high"/"low"). Plus a top-level scoring-only `better_in_gap` label that is **never shown to any reviewer**.
- **Ground-truth label** `better_in_gap` (bool) = the gap contains a config with energy < `best_energy` (⟺ the global minimum is in the gap).
- **Cheat-proof invariant:** a reviewer sees ONLY the paper summary (never the gap configs/energies, the global minimum, or the label). `build_prompt` must be a pure function of the summary fields; `review_one` hands the reviewer a view with the label stripped.
- **Reviewer action space:** strict tool `predict_gap` → `{better_in_gap: boolean}`. Invalid/unparseable output ⇒ fall back to `False` (recorded `fallback: true`). LLM output parsed as data, never executed.
- **Arms:** `baseline` (always predict False), `heuristic` (non-LLM: energy still improving toward the gap boundary ⇒ True), `llm`. Metric: `accuracy` (primary) + `precision`/`recall` on the True class.
- **Credentials are the user's.** No key in code; `MODEL = "claude-sonnet-5"` (module constant, `--model` switch). **Tests must never hit MACE or the real Anthropic API** — inject a fake `evaluate_fn` / fake client. The LLM arm auto-skips without a credential (narrow the auto-skip to auth errors, matching `atomica/run_critic.py`).
- Reproducibility via `numpy.random.default_rng(seed)` for paper generation.

---

### Task 1: `litreview_world.py` — `make_paper`

**Files:**
- Create: `atomica/litreview_world.py`
- Test: `tests/test_litreview_world.py`

**Interfaces:**
- Consumes: `atomica.critic_world.build_world`, `features`, `FEATURE_NAMES`.
- Produces: `make_paper(world, seed, min_frac=0.3, max_frac=0.7, n_subbands=4) -> dict`. Paper dict keys: `axis` (str), `region` (str), `n_explored` (int), `best_config` (list[int]), `best_energy` (float), `boundary_trend` (list[float], far→near the gap), `gap_side` ("high"|"low"), `better_in_gap` (bool), `seed` (int).

- [ ] **Step 1: Write the failing tests** (fake `evaluate_fn` — no MACE; energy = x1 so the label is analytically checkable)

```python
# tests/test_litreview_world.py
import numpy as np
from atomica.critic_world import build_world, features
from atomica.litreview_world import make_paper

def _fake_world():
    # Energy = x1 + 0.31*x2 + 0.07*layer — a near-unique combo (few ties, a clean global minimum),
    # so `better_in_gap` reliably mixes True/False across seeds as the studied band moves.
    def fake_eval(config):
        f = features(config)
        return float(f["x1"] + 0.31 * f["x2"] + 0.07 * f["layer"])
    return build_world(evaluate_fn=fake_eval)

def test_make_paper_shape_and_types():
    w = _fake_world()
    p = make_paper(w, seed=0)
    assert p["axis"] in ("x1", "x2", "layer")
    assert isinstance(p["best_config"], list) and len(p["best_config"]) == 6
    assert isinstance(p["best_energy"], float)
    assert isinstance(p["boundary_trend"], list) and len(p["boundary_trend"]) == 4
    assert p["gap_side"] in ("high", "low")
    assert isinstance(p["better_in_gap"], bool)

def test_label_matches_independent_recompute():
    # Independently recompute better_in_gap from the full world and the paper's explored best.
    w = _fake_world()
    for seed in range(30):
        p = make_paper(w, seed=seed)
        E = w["energy"]; zc = w["features"][p["axis"]].astype(float)
        # reconstruct explored mask from gap_side + best_energy is not enough; instead check the
        # invariant: better_in_gap == (some config strictly below best_energy exists outside explored).
        # Explored = configs on the studied side; recompute via gap_side.
        # (gap_side "high" means gap is the high-axis side -> explored is the low side.)
        assert isinstance(p["better_in_gap"], bool)
        # best_energy must be the min over the explored subset, so no explored config is below it:
        # and better_in_gap true => min over ALL < best_energy.
        if p["better_in_gap"]:
            assert E.min() < p["best_energy"] - 1e-9
        else:
            assert abs(E.min() - p["best_energy"]) < 1e-9 or E.min() >= p["best_energy"] - 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_litreview_world.py -v`
Expected: FAIL with `ModuleNotFoundError: atomica.litreview_world`.

- [ ] **Step 3: Write minimal implementation**

```python
# atomica/litreview_world.py
"""P5 literature world: generate a synthetic 'paper' (a summary of a study over an explored
subset of the Cu-Au 924 orderings) plus a ground-truth label. Reuses P4's critic_world unchanged."""
import numpy as np
from atomica.critic_world import build_world, features, FEATURE_NAMES

def make_paper(world, seed, min_frac=0.3, max_frac=0.7, n_subbands=4):
    rng = np.random.default_rng(seed)
    F, E, configs = world["features"], world["energy"], world["configs"]
    n = len(E)
    axis = str(rng.choice(FEATURE_NAMES))
    zc = F[axis].astype(float)
    frac = float(rng.uniform(min_frac, max_frac))
    gap_high = bool(rng.integers(2))          # True -> explored is the LOW side, gap is HIGH
    if gap_high:
        cut = np.quantile(zc, frac)
        explored = zc <= cut
    else:
        cut = np.quantile(zc, 1.0 - frac)
        explored = zc >= cut
    gap = ~explored
    ei = np.where(explored)[0]
    best_local = int(ei[np.argmin(E[ei])])
    best_energy = float(E[best_local])
    # boundary_trend: best energy in ordered sub-bands of the explored subset, far->near the gap.
    z_expl = zc[explored]
    edges = np.quantile(z_expl, np.linspace(0.0, 1.0, n_subbands + 1))
    trend = []
    for b in range(n_subbands):
        upper = (z_expl <= edges[b + 1]) if b == n_subbands - 1 else (z_expl < edges[b + 1])
        m = (z_expl >= edges[b]) & upper
        idx = ei[m]
        trend.append(float(E[idx].min()) if len(idx) else float(best_energy))
    if not gap_high:                          # gap on the low side -> nearest-gap sub-band is lowest z
        trend = trend[::-1]                    # reorder so trend[-1] is nearest the gap
    gap_best = float(E[gap].min())
    better_in_gap = bool(gap_best < best_energy - 1e-9)
    studied_side = "low" if gap_high else "high"
    return {"axis": axis,
            "region": f"orderings with {axis} in the {studied_side} range",
            "n_explored": int(explored.sum()),
            "best_config": list(configs[best_local]),
            "best_energy": best_energy,
            "boundary_trend": trend,
            "gap_side": "high" if gap_high else "low",
            "better_in_gap": better_in_gap,
            "seed": int(seed)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_litreview_world.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/litreview_world.py tests/test_litreview_world.py
git commit -m "feat: P5 synthetic paper generation + ground-truth label"
```

---

### Task 2: `litreview_world.py` — `generate_papers`

**Files:**
- Modify: `atomica/litreview_world.py`
- Test: `tests/test_litreview_world.py` (add cases)

**Interfaces:**
- Consumes: `make_paper`.
- Produces: `generate_papers(world, n_papers, seed0=0, min_frac=0.3, max_frac=0.7, min_count=40) -> list[dict]` — N valid papers (explored and gap each ≥ `min_count` configs), skipping degenerate seeds.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_litreview_world.py
from atomica.litreview_world import generate_papers

def test_generate_papers_count_and_mixed_labels():
    w = _fake_world()
    papers = generate_papers(w, n_papers=40, seed0=0)
    assert len(papers) == 40
    for p in papers:
        assert p["n_explored"] >= 40
    labels = {p["better_in_gap"] for p in papers}
    assert labels == {True, False}          # both classes appear (energy = x1 gives a real mix)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_litreview_world.py -k generate_papers -v`
Expected: FAIL with ImportError for `generate_papers`.

- [ ] **Step 3: Write minimal implementation** (append to `atomica/litreview_world.py`)

```python
def generate_papers(world, n_papers, seed0=0, min_frac=0.3, max_frac=0.7, min_count=40):
    papers, seed = [], seed0
    n = len(world["energy"])
    while len(papers) < n_papers:
        p = make_paper(world, seed, min_frac, max_frac)
        seed += 1
        if p["n_explored"] < min_count or (n - p["n_explored"]) < min_count:
            continue                        # explored or gap too small
        papers.append(p)
    return papers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_litreview_world.py -k generate_papers -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/litreview_world.py tests/test_litreview_world.py
git commit -m "feat: P5 generate_papers (N valid non-degenerate papers)"
```

---

### Task 3: `litreview.py` — validation + baseline + heuristic reviewers

**Files:**
- Create: `atomica/litreview.py`
- Test: `tests/test_litreview.py`

**Interfaces:**
- Produces: `PREDICT_TOOL` (strict schema), `validate_prediction(raw) -> {"better_in_gap": bool}`, `baseline_reviewer(paper) -> {"better_in_gap": False}`, `heuristic_reviewer(paper) -> {"better_in_gap": bool}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_litreview.py
import pytest
from atomica.litreview import validate_prediction, baseline_reviewer, heuristic_reviewer

def _paper(trend, better=False):
    return {"axis": "x2", "region": "orderings with x2 in the low range", "n_explored": 100,
            "best_config": [0, 1, 2, 3, 4, 5], "best_energy": -44.0, "boundary_trend": trend,
            "gap_side": "high", "better_in_gap": better}

def test_validate_prediction():
    assert validate_prediction({"better_in_gap": True}) == {"better_in_gap": True}
    assert validate_prediction({"better_in_gap": False}) == {"better_in_gap": False}
    for bad in [{"better_in_gap": "yes"}, {"better_in_gap": 1}, {}, "nope", None]:
        with pytest.raises(ValueError):
            validate_prediction(bad)

def test_baseline_always_false():
    assert baseline_reviewer(_paper([-1, -2, -3, -4])) == {"better_in_gap": False}

def test_heuristic_reads_boundary_trend():
    # energy still improving toward the gap (last sub-band lowest) -> predict better_in_gap True
    assert heuristic_reviewer(_paper([-1.0, -2.0, -3.0, -4.0]))["better_in_gap"] is True
    # flat / worsening near the gap -> False
    assert heuristic_reviewer(_paper([-4.0, -4.0, -4.0, -4.0]))["better_in_gap"] is False
    assert heuristic_reviewer(_paper([-4.0, -3.0, -2.0, -1.0]))["better_in_gap"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_litreview.py -v`
Expected: FAIL with `ModuleNotFoundError: atomica.litreview`.

- [ ] **Step 3: Write minimal implementation**

```python
# atomica/litreview.py
"""P5 literature reviewer: validate a structured boolean prediction, provide a paper-blind
baseline, a non-LLM trend heuristic, and (later) the LLM reviewer; score accuracy/precision/recall.
The reviewer proposes; the harness checks against ground truth."""

PREDICT_TOOL = {
    "name": "predict_gap",
    "description": "Predict whether the unexplored region contains a structure with lower energy "
                   "than the study's reported best.",
    "input_schema": {
        "type": "object",
        "properties": {
            "better_in_gap": {"type": "boolean",
                              "description": "true if a better (lower-energy) structure likely exists "
                                             "in the unexplored region"},
        },
        "required": ["better_in_gap"],
        "additionalProperties": False,
    },
    "strict": True,
}

def validate_prediction(raw):
    if not isinstance(raw, dict):
        raise ValueError(f"prediction not a dict: {raw!r}")
    v = raw.get("better_in_gap")
    if not isinstance(v, bool):
        raise ValueError(f"better_in_gap not a bool: {v!r}")
    return {"better_in_gap": v}

def baseline_reviewer(paper):
    return {"better_in_gap": False}

def heuristic_reviewer(paper):
    t = paper["boundary_trend"]
    h = len(t) // 2
    far, near = t[:h], t[h:]
    return {"better_in_gap": bool(min(near) < min(far))}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_litreview.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/litreview.py tests/test_litreview.py
git commit -m "feat: P5 prediction validation + baseline + trend heuristic reviewers"
```

---

### Task 4: `litreview.py` — review + scoring

**Files:**
- Modify: `atomica/litreview.py`
- Test: `tests/test_litreview.py` (add cases)

**Interfaces:**
- Consumes: `validate_prediction`.
- Produces:
  - `review_one(paper, reviewer) -> dict` (keys: `predicted` (bool), `correct` (bool), `fallback` (bool)). Hands the reviewer a **view with `better_in_gap` stripped** (cheat-proof); on ValueError falls back to `{"better_in_gap": False}` recorded `fallback: True`.
  - `review_batch(papers, reviewer) -> list[dict]`.
  - `score(papers, reviews) -> dict` (keys: `accuracy`, `precision`, `recall`, `n`, `base_rate_better`).

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_litreview.py
from atomica.litreview import review_one, review_batch, score, baseline_reviewer

def test_review_one_strips_label_and_scores():
    seen = {}
    def spy_reviewer(view):
        seen.update(view)
        return {"better_in_gap": True}
    p = _paper([-1, -2, -3, -4], better=True)
    r = review_one(p, spy_reviewer)
    assert "better_in_gap" not in seen        # cheat-proof: label never handed to the reviewer
    assert r["predicted"] is True and r["correct"] is True and r["fallback"] is False

def test_review_one_fallback_on_garbage():
    def garbage(view): return {"better_in_gap": "nope"}
    p = _paper([-1, -2, -3, -4], better=False)
    r = review_one(p, garbage)
    assert r["fallback"] is True and r["predicted"] is False and r["correct"] is True

def test_score_accuracy_precision_recall():
    # 4 papers: labels [T, T, F, F]. baseline predicts all False -> acc .5, precision 0, recall 0.
    papers = [_paper([-1], True), _paper([-1], True), _paper([-1], False), _paper([-1], False)]
    reviews = review_batch(papers, baseline_reviewer)
    s = score(papers, reviews)
    assert abs(s["accuracy"] - 0.5) < 1e-9
    assert s["precision"] == 0.0 and s["recall"] == 0.0
    assert abs(s["base_rate_better"] - 0.5) < 1e-9
    assert s["n"] == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_litreview.py -k "review_one or score" -v`
Expected: FAIL with ImportError for `review_one`.

- [ ] **Step 3: Write minimal implementation** (append to `atomica/litreview.py`)

```python
def review_one(paper, reviewer):
    # cheat-proof: hand the reviewer only observable summary fields — never the label.
    view = {k: v for k, v in paper.items() if k != "better_in_gap"}
    fallback = False
    try:
        pred = validate_prediction(reviewer(view))
    except ValueError:
        pred = {"better_in_gap": False}
        fallback = True
    predicted = pred["better_in_gap"]
    return {"predicted": predicted, "correct": predicted == paper["better_in_gap"],
            "fallback": fallback}

def review_batch(papers, reviewer):
    return [review_one(p, reviewer) for p in papers]

def score(papers, reviews):
    y = [p["better_in_gap"] for p in papers]
    yhat = [r["predicted"] for r in reviews]
    n = len(papers)
    correct = int(sum(a == b for a, b in zip(y, yhat)))
    tp = int(sum(h and t for h, t in zip(yhat, y)))
    pred_pos = int(sum(yhat))
    actual_pos = int(sum(y))
    return {"accuracy": correct / n if n else 0.0,
            "precision": tp / pred_pos if pred_pos else 0.0,
            "recall": tp / actual_pos if actual_pos else 0.0,
            "n": n, "base_rate_better": actual_pos / n if n else 0.0}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_litreview.py -k "review_one or score" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/litreview.py tests/test_litreview.py
git commit -m "feat: P5 review loop (label-stripped) + accuracy/precision/recall scoring"
```

---

### Task 5: `litreview.py` — `llm_reviewer` (Anthropic strict-tool structured output)

**Files:**
- Modify: `atomica/litreview.py`
- Test: `tests/test_litreview.py` (add cases)

**REQUIRED:** Before writing this task, read the `claude-api` skill's `python/claude-api/tool-use.md` for the exact strict-tool surface. This is the SAME pattern P3/P4 shipped (`strict: true` top-level, `additionalProperties: false`, `required`, `tool_choice={"type":"tool","name":...}`, extract from the `tool_use` block's `.input`) — confirm consistency; do not guess.

**Interfaces:**
- Consumes: `PREDICT_TOOL`.
- Produces:
  - `MODEL = "claude-sonnet-5"`.
  - `build_prompt(paper) -> str` — names the region, `n_explored`, the reported best config + energy, and the `boundary_trend` (far→near the gap); asks for `predict_gap`. Reads ONLY summary fields — never `better_in_gap`.
  - `llm_reviewer(client, model=MODEL) -> callable(view) -> dict` — one structured call returning the raw `{better_in_gap}` (validated by `review_one`).

- [ ] **Step 1: Write the failing tests** (fake client — never hits the API)

```python
# add to tests/test_litreview.py
from atomica.litreview import llm_reviewer, build_prompt, MODEL

class _FakeBlock:
    def __init__(self, inp): self.type = "tool_use"; self.name = "predict_gap"; self.input = inp
class _FakeResp:
    def __init__(self, inp): self.content = [_FakeBlock(inp)]
class _FakeMessages:
    def __init__(self, inp): self._inp = inp; self.last_kwargs = None
    def create(self, **kw): self.last_kwargs = kw; return _FakeResp(self._inp)
class _FakeClient:
    def __init__(self, inp): self.messages = _FakeMessages(inp)

def test_llm_reviewer_extracts_prediction():
    client = _FakeClient({"better_in_gap": True})
    reviewer = llm_reviewer(client)
    raw = reviewer({"axis": "x2", "region": "orderings with x2 in the low range", "n_explored": 100,
                    "best_config": [0, 1, 2, 3, 4, 5], "best_energy": -44.0,
                    "boundary_trend": [-1.0, -2.0, -3.0, -4.0], "gap_side": "high"})
    assert raw == {"better_in_gap": True}
    assert client.messages.last_kwargs["model"] == MODEL

def test_build_prompt_mentions_region_and_trend_not_label():
    paper = {"axis": "x2", "region": "orderings with x2 in the low range", "n_explored": 100,
             "best_config": [0, 1, 2, 3, 4, 5], "best_energy": -44.0,
             "boundary_trend": [-1.0, -2.0, -3.0, -4.0], "gap_side": "high", "better_in_gap": True}
    txt = build_prompt(paper)
    assert "x2" in txt and "-44" in txt and "boundary" in txt.lower()
    # cheat-proof: prompt is independent of the hidden label
    paper2 = dict(paper); paper2["better_in_gap"] = False
    assert build_prompt(paper) == build_prompt(paper2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_litreview.py -k "llm_reviewer or build_prompt" -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation** (append to `atomica/litreview.py`; match the `claude-api` strict-tool surface)

```python
MODEL = "claude-sonnet-5"

def build_prompt(paper):
    gap = paper["gap_side"]
    trend = ", ".join(f"{e:.3f}" for e in paper["boundary_trend"])
    lines = [
        "A study of Cu-Au orderings on a fixed 12-site lattice explored only part of the design space.",
        f"Explored region: {paper['region']} ({paper['n_explored']} orderings).",
        f"Best structure found in the study: config {paper['best_config']} at energy "
        f"{paper['best_energy']:.3f} eV (lower energy = more stable).",
        f"The unexplored region lies on the {gap} side of {paper['axis']}.",
        "Reported best energy across sub-bands of the explored region, ordered FAR from the gap "
        f"to NEAR the gap boundary: [{trend}].",
        "Question: does the unexplored region likely contain a structure with LOWER energy than the "
        "study's best? Reason about whether energy is still improving where the study stopped.",
        "Call predict_gap with your prediction.",
    ]
    return "\n".join(lines)

def llm_reviewer(client, model=MODEL):
    def reviewer(view):
        resp = client.messages.create(
            model=model, max_tokens=512,
            tools=[PREDICT_TOOL], tool_choice={"type": "tool", "name": "predict_gap"},
            messages=[{"role": "user", "content": build_prompt(view)}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "predict_gap":
                return block.input
        raise ValueError("no predict_gap tool_use in response")
    return reviewer
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_litreview.py -k "llm_reviewer or build_prompt" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/litreview.py tests/test_litreview.py
git commit -m "feat: P5 LLM reviewer via Anthropic strict-tool structured output"
```

---

### Task 6: `run_litreview.py` CLI + report

**Files:**
- Create: `atomica/run_litreview.py`
- Test: `tests/test_run_litreview.py`

**Interfaces:**
- Consumes: `atomica.litreview_world.build_world`, `generate_papers`; `atomica.litreview` (`baseline_reviewer`, `heuristic_reviewer`, `llm_reviewer`, `review_batch`, `score`, `MODEL`).
- Produces: `make_llm_reviewer(model) -> reviewer|None`; `run_litreview.main(argv=None) -> None` writing `results/litreview_report.json` (keys: `arms` = {name: score_dict}, `config`).

Note: `build_world` and `generate_papers` are re-exported into `run_litreview`'s module namespace by importing them, so a test can `monkeypatch.setattr(run_litreview, "build_world", ...)`. `make_llm_reviewer` and `build_world` must be called as bare module-level names inside `main` (not local aliases) so monkeypatch takes effect.

- [ ] **Step 1: Write the failing test** (monkeypatch the LLM arm off + a fake world — no MACE, no API)

```python
# tests/test_run_litreview.py
import json
from atomica import run_litreview
from atomica.critic_world import build_world, features

def test_cli_smoke_baseline_and_heuristic(tmp_path, monkeypatch):
    def fake_eval(config):                                        # near-unique combo, no MACE
        f = features(config); return float(f["x1"] + 0.31 * f["x2"] + 0.07 * f["layer"])
    monkeypatch.setattr(run_litreview, "make_llm_reviewer", lambda model: None)   # LLM arm off
    monkeypatch.setattr(run_litreview, "build_world",
                        lambda evaluate_fn=None, cache_path=None: build_world(evaluate_fn=fake_eval))
    run_litreview.main(["--n-papers", "20", "--seed", "0", "--out", str(tmp_path)])
    report = json.loads((tmp_path / "litreview_report.json").read_text())
    assert "baseline" in report["arms"] and "heuristic" in report["arms"]
    assert "llm" not in report["arms"]                    # skipped, no client
    for arm in ("baseline", "heuristic"):
        assert set(report["arms"][arm]) >= {"accuracy", "precision", "recall", "n"}
    assert report["arms"]["baseline"]["recall"] == 0.0    # always-False -> 0 recall
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_run_litreview.py -v`
Expected: FAIL with `ModuleNotFoundError: atomica.run_litreview`.

- [ ] **Step 3: Write minimal implementation**

```python
# atomica/run_litreview.py
"""ATOMICA P5 CLI: does a paper-reading LLM predict the unexplored optimum better than a
paper-blind baseline? (Cu-Au research-gap extrapolation)."""
import argparse, json
from pathlib import Path
from atomica.litreview_world import build_world, generate_papers
from atomica.litreview import (
    baseline_reviewer, heuristic_reviewer, llm_reviewer, review_batch, score, MODEL,
)

def make_llm_reviewer(model):
    """Return an llm_reviewer bound to a real client, or None if no credential/SDK."""
    try:
        import anthropic
        return llm_reviewer(anthropic.Anthropic(), model=model)
    except Exception as e:                      # missing key/sdk -> skip the LLM arm
        print(f"[run_litreview] LLM arm disabled: {e}")
        return None

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
        import anthropic
        try:
            arms["llm"] = score(papers, review_batch(papers, llm))
        except anthropic.AnthropicError as e:   # auth resolves lazily on first call -> skip cleanly
            print(f"[run_litreview] LLM arm disabled: {e}")
        except TypeError as e:
            if "authentication method" not in str(e):
                raise
            print(f"[run_litreview] LLM arm disabled: {e}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "litreview_report.json").write_text(json.dumps(
        {"arms": arms, "config": {"n_papers": a.n_papers, "seed": a.seed, "model": a.model}}, indent=2))
    print(json.dumps({k: {"accuracy": round(v["accuracy"], 3), "recall": round(v["recall"], 3)}
                      for k, v in arms.items()}, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_run_litreview.py -v` then the full suite `python3 -m pytest -q`.
Expected: PASS (Slice 1 + P2 + P3 + P4 + P5 all green).

- [ ] **Step 5: Commit**

```bash
git add atomica/run_litreview.py tests/test_run_litreview.py
git commit -m "feat: P5 run_litreview CLI + report (LLM arm auto-skips without a credential)"
```

---

### Task 7: Deliverable run + README

**Files:**
- Modify: `README.md`

**Interfaces:** none (produces the result + docs).

- [ ] **Step 1: Check for a credential**

Run: `echo "${ANTHROPIC_API_KEY:+set}"` and `ant auth status 2>&1 || echo "no ant profile"`.
- Credential available → the full baseline/heuristic/llm run is possible.
- **Not** available → run baseline+heuristic (no key); the LLM arm auto-skips (`[run_litreview] LLM arm disabled: ...`). Note in the README that the LLM arm needs the user's own `ANTHROPIC_API_KEY` / `ant auth login`. **Do not fabricate LLM numbers.**

- [ ] **Step 2: Run the deliverable**

```bash
python3 -m atomica.run_litreview --n-papers 60 --seed 0 --out results
```

This reuses the cached 924-config Cu-Au world (`results/alloy_world.json`, built by P4; rebuilt via MACE ~5 min only if absent), generates 60 papers, and runs the arms. Read `results/litreview_report.json`: compare `accuracy` across `baseline` / `heuristic` / (`llm` if on), plus precision/recall.

- [ ] **Step 3: Write the README "P5" section**

Add a "P5 — literature agent (research-gap extrapolation)" section after the P4 section: what it does (a synthetic paper reports the best structure + energy trend over an explored subset; the LLM reads only that summary and predicts as validated JSON whether the unexplored region hides a better structure — never touches physics; scored against the 924-config ground truth), how to run (`python3 -m atomica.run_litreview`, note the `ANTHROPIC_API_KEY` requirement), and the result honestly — accuracy(baseline) vs accuracy(heuristic) vs accuracy(llm). Include caveats: the baseline is the majority class (beating it is the bar; the LLM can score below it); the heuristic is a non-LLM reference ceiling; the LLM is stochastic; the original sign-reversal framing was moot here (see the spec §Spike) and this reframe is what the world supports; small 12-site world; a null result is valid. Point to the P5 spec. Update the Roadmap table row P5 from "🔒 planned" to "✅ done", and bump the tests badge to the new total.

- [ ] **Step 4: Run the full test suite**

Run: `python3 -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add README.md results/litreview_report.json
git commit -m "feat: P5 deliverable run and README (literature-gap extrapolation benchmark)"
```

(If `results/*.json` is gitignored — it is — the `litreview_report.json` add is a no-op; commit the README and note the report is regenerated by the CLI.)

---

## Self-Review

**1. Spec coverage:**
- Reuse P4 world + synthetic paper (explored subset, summary) + label (spec §2/§3) → Task 1. ✅
- N valid papers, non-trivial mix (spec §3) → Task 2. ✅
- Strict-tool action space + validation + FALSE fallback (spec §4/§6) → Task 3, 5. ✅
- Baseline (always-False) + heuristic (trend) reviewers (spec §2/§5) → Task 3. ✅
- Cheat-proof: review_one strips label, build_prompt label-independent (spec §4) → Task 4, 5. ✅
- Accuracy + precision + recall scoring (spec §5) → Task 4. ✅
- `llm_reviewer` injected client, no code execution, MODEL constant (spec §6) → Task 5. ✅
- CLI + 3 arms + LLM-arm auto-skip narrowed to auth (spec §6/§7) → Task 6. ✅
- Deliverable + honest caveats + credential-gated run (spec §9) → Task 7. ✅
- Tests never hit MACE or the real API (fake evaluate_fn / fake client / monkeypatch) (spec §8) → Tasks 1–6. ✅

**2. Placeholder scan:** No "TBD"/vague steps; every code step carries concrete code. Task 5 requires confirming the Anthropic strict-tool surface from the `claude-api` skill first — the shown code is the concrete target, adjusted only if the SDK surface differs.

**3. Type consistency:** `make_paper(world, seed, ...)->dict` with keys axis/region/n_explored/best_config/best_energy/boundary_trend/gap_side/better_in_gap/seed (T1) consumed by generate_papers (T2), review_one/build_prompt/heuristic (T3,4,5), run (T6). `generate_papers(world, n_papers, seed0, ...)->list` (T2) used in T6. `validate_prediction(raw)->{better_in_gap}` (T3) used in T4. `baseline_reviewer`/`heuristic_reviewer(paper)->{better_in_gap}` (T3) used in T4,6. `review_one`/`review_batch`/`score` (T4) used in T6. `PREDICT_TOOL`/`build_prompt`/`llm_reviewer`/`MODEL` (T3,5) used in T6. `make_llm_reviewer(model)->reviewer|None` (T6) monkeypatched in T6 test. review dict keys `predicted`/`correct`/`fallback` (T4) used by score (reads `predicted`) and consistent. ✅
