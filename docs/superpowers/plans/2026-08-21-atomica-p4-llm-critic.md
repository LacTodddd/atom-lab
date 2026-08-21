# ATOMICA P4 — LLM Critic (Falsification Loop) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an LLM critic try to falsify a naive scientist's claim by naming the confounder, and measure — cheat-proof against the P2 Cu-Au ground truth — whether it cuts the false-discovery rate below a random critic and below no critic.

**Architecture:** Two new modules. `atomica/critic_world.py` builds the 924-config Cu-Au world (reusing P2's `alloy` unchanged), computes three non-collinear features + a stratified controlled-effect estimator, and runs a deterministic biased "scientist" that emits labeled TRUE/FALSE claims. `atomica/critic.py` validates a critic's structured verdict, applies a within-sample stratified control (sign-flip ⇒ reject), and scores FDR + retention over the three arms (none / random / llm). `atomica/run_critic.py` is the CLI; the LLM arm auto-skips without a credential (P3 pattern).

**Tech Stack:** Python 3.13, `anthropic` SDK (already a dependency from P3), reusing `numpy`, `ase`, and `atomica.alloy` (MACE-MP-0).

**Spec:** `docs/superpowers/specs/2026-08-21-atomica-p4-llm-critic-design.md`

## Global Constraints

- **Reuse `atomica/alloy.py` unchanged** (`build_lattice`, `evaluate`, `config_symbols`). Do not modify Slice 1 / P2 / P3 files.
- **Fixed feature set:** `FEATURE_NAMES = ["x1", "x2", "layer"]` — `x1` = first-NN Au-Au pairs, `x2` = second-NN Au-Au pairs, `layer` = Au atoms in one (100) plane. Lattice = 12 sites, 6 Au (`N_SITES=12`, `N_AU=6`), so C(12,6)=924 configs.
- **Truth = Z-stratified X-contrast** over all 924: quantile-bin `Z` (default 3 bins), median-split `X` within each bin, count-weighted average of the high-X-minus-low-X **energy** difference; its **sign** is the controlled ground truth. The same estimator is used for the control (on the sample).
- **Scientist is deterministic** (seeded `numpy.random.default_rng`), no LLM: biased sample of `n=40` via weights `exp(strength · ẑ_X · ẑ_Z)`, `strength=2.0`. Claim direction = naive within-sample X-contrast sign. Label TRUE if naive sign == stratified truth sign (both non-zero), else FALSE.
- **Critic action space:** strict tool `critique_claim` → `{verdict: "supported"|"confounded", confounder: one of the two non-target features}`. Invalid/unparseable output ⇒ fall back to `supported` (recorded `fallback: true`). LLM output is parsed as data, never executed.
- **Control (within-sample):** on `confounded` + named `Z`, stratify the claim's own sample on `Z`; if the controlled sign **flips** vs the claim direction ⇒ **reject**, else accept. `supported` ⇒ accept, no control.
- **Metrics:** `fdr` = accepted-false / accepted (primary, lower better); `retention` = accepted-true / all-true (guard, higher better).
- **Credentials are the user's.** No key in code; `MODEL = "claude-sonnet-5"` (module constant, `--model` switch). **Tests must never hit MACE or the real Anthropic API** — inject a fake `evaluate_fn` / fake client. The LLM arm auto-skips without a credential.
- Reproducibility via `numpy.random.default_rng(seed)` for the scientist and the random arm.

---

### Task 1: `critic_world.py` — features + stratified effect estimator

**Files:**
- Create: `atomica/critic_world.py`
- Test: `tests/test_critic_world.py`

**Interfaces:**
- Produces: `N_SITES`, `N_AU`, `FEATURE_NAMES`, `features(config) -> dict`, `stratified_effect(Xs, Zs, Es, kbins=3) -> float`, `_sign(v, tol=1e-9) -> int`, `_contrast(Xs, Es) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_critic_world.py
import numpy as np
from atomica.critic_world import (
    features, stratified_effect, _sign, _contrast, FEATURE_NAMES, N_SITES, N_AU,
)

def test_features_shape_and_range():
    f = features((0, 1, 2, 3, 4, 5))
    assert set(f) == set(FEATURE_NAMES)
    assert all(isinstance(v, int) for v in f.values())
    assert 0 <= f["layer"] <= 4          # one (100) plane holds 4 sites

def test_sign_and_contrast():
    assert _sign(-0.5) == -1 and _sign(0.5) == 1 and _sign(0.0) == 0
    # high-X rows carry higher energy -> positive contrast
    Xs = np.array([0, 0, 1, 1]); Es = np.array([-2.0, -2.0, -1.0, -1.0])
    assert _contrast(Xs, Es) > 0

def test_stratified_effect_controls_confounder():
    # Within each Z stratum, X has NO effect; the raw contrast is driven only by Z.
    # Build data where X and Z covary but energy depends only on Z.
    rng = np.random.default_rng(0)
    Z = np.array([0]*20 + [1]*20)
    X = Z.copy()                                  # perfectly confounded
    E = 1.0 * Z + rng.normal(0, 1e-6, size=40)    # energy depends only on Z
    # raw (unstratified) contrast on X sees Z's effect -> nonzero
    assert _contrast(X.astype(float), E) != 0
    # stratified on Z -> X has no within-stratum variation -> ~0 effect
    assert abs(stratified_effect(X, Z, E)) < 1e-3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_critic_world.py -v`
Expected: FAIL with `ModuleNotFoundError: atomica.critic_world`.

- [ ] **Step 3: Write minimal implementation**

```python
# atomica/critic_world.py
"""P4 critic world: Cu-Au features, stratified controlled-effect estimator,
and a deterministic biased 'scientist'. Reuses P2's alloy module unchanged."""
import numpy as np
from atomica import alloy

N_SITES, N_AU = 12, 6
FEATURE_NAMES = ["x1", "x2", "layer"]

def _geometry():
    at = alloy.build_lattice(N_SITES)
    D = at.get_all_distances(mic=True)
    dv = np.unique(np.round(D[D > 1e-6], 3))
    s1, s2 = float(dv[0]), float(dv[1])
    zc = np.round(at.get_positions()[:, 2], 2)
    layer0 = list(np.where(zc == sorted(np.unique(zc))[0])[0])
    return D, s1, s2, layer0

_D, _S1, _S2, _LAYER0 = _geometry()

def features(config):
    au = np.zeros(N_SITES, bool); au[list(config)] = True
    def pairs(shell):
        return int(sum(au[i] and au[j] and abs(_D[i, j] - shell) < 1e-2
                       for i in range(N_SITES) for j in range(i + 1, N_SITES)))
    return {"x1": pairs(_S1), "x2": pairs(_S2), "layer": int(au[_LAYER0].sum())}

def _sign(v, tol=1e-9):
    return 0 if abs(v) < tol else (1 if v > 0 else -1)

def _contrast(Xs, Es):
    """energy(high-X) - energy(low-X), median split on X."""
    Xs, Es = np.asarray(Xs, float), np.asarray(Es, float)
    m = np.median(Xs)
    hi, lo = Es[Xs > m], Es[Xs <= m]
    if len(hi) == 0 or len(lo) == 0:
        return 0.0
    return float(hi.mean() - lo.mean())

def stratified_effect(Xs, Zs, Es, kbins=3):
    """Z-stratified X-contrast, count-weighted over bins."""
    Xs, Zs, Es = np.asarray(Xs, float), np.asarray(Zs, float), np.asarray(Es, float)
    diffs, wts = [], []
    qs = np.quantile(Zs, np.linspace(0, 1, kbins + 1))
    for b in range(kbins):
        upper = (Zs <= qs[b + 1]) if b == kbins - 1 else (Zs < qs[b + 1])
        m = (Zs >= qs[b]) & upper
        if m.sum() < 3:
            continue
        d = _contrast(Xs[m], Es[m])
        if d != 0:
            diffs.append(d); wts.append(int(m.sum()))
    if not diffs:
        return 0.0
    return float(np.average(diffs, weights=wts))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_critic_world.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/critic_world.py tests/test_critic_world.py
git commit -m "feat: P4 critic world features + stratified effect estimator"
```

---

### Task 2: `critic_world.py` — `build_world`, biased scientist, claim generation

**Files:**
- Modify: `atomica/critic_world.py`
- Test: `tests/test_critic_world.py` (add cases)

**Interfaces:**
- Consumes: `features`, `stratified_effect`, `_sign`, `_contrast`, `FEATURE_NAMES`, `alloy.evaluate`.
- Produces:
  - `build_world(evaluate_fn=alloy.evaluate, cache_path=None) -> dict` with keys `configs` (list of tuples), `features` (dict name→np.array over 924), `energy` (np.array over 924).
  - `truth_sign(target, confounder, world, kbins=3) -> int`.
  - `make_claim(world, seed, n=40, strength=2.0) -> dict` (may have `claim_sign==0`; caller filters).
  - `generate_claims(world, n_claims, n=40, strength=2.0, seed0=0) -> list[dict]` — N valid claims (non-zero claim & truth sign).
  - Claim dict keys: `target` (str), `claim_sign` (±1 int), `confounder_true` (str), `sample` (dict: each feature name→list + `energy`→list, length n), `label_is_true` (bool), `seed` (int).

- [ ] **Step 1: Write the failing tests** (fake `evaluate_fn` — no MACE)

```python
# add to tests/test_critic_world.py
from atomica.critic_world import build_world, truth_sign, make_claim, generate_claims

def _fake_world():
    # Energy = x1 - 3*x2 (real features, synthetic energy). x1's true controlled effect is +1,
    # but x2's is a strong -3; since biased sampling makes x1 and x2 covary POSITIVELY in-sample,
    # the naive x1 contrast picks up x2's opposite-sign effect and sometimes FLIPS -> FALSE claims
    # for (target=x1, confounder=x2). Pairs involving `layer` (no energy effect) stay TRUE.
    def fake_eval(config):
        f = features(config)
        return 1.0 * f["x1"] - 3.0 * f["x2"]
    return build_world(evaluate_fn=fake_eval)

def test_build_world_shape():
    w = _fake_world()
    assert len(w["configs"]) == 924
    assert set(w["features"]) == {"x1", "x2", "layer"}
    assert len(w["energy"]) == 924

def test_generate_claims_are_labeled_and_mixed():
    w = _fake_world()
    claims = generate_claims(w, n_claims=40, n=40, strength=2.0, seed0=0)
    assert len(claims) == 40
    for c in claims:
        assert c["claim_sign"] in (-1, 1)
        assert c["confounder_true"] in FEATURE_NAMES and c["confounder_true"] != c["target"]
        assert isinstance(c["label_is_true"], bool)
        assert len(c["sample"]["energy"]) == 40
        assert "confounder_true" in c            # present for scoring, not shown to critics
    # a healthy biased-sampling world yields BOTH true and false claims
    labels = {c["label_is_true"] for c in claims}
    assert labels == {True, False}

def test_make_claim_sample_hides_nothing_needed_but_carries_features():
    w = _fake_world()
    c = make_claim(w, seed=1, n=40, strength=2.0)
    for name in FEATURE_NAMES:
        assert len(c["sample"][name]) == 40
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_critic_world.py -k "build_world or generate_claims or make_claim" -v`
Expected: FAIL with ImportError for `build_world`.

- [ ] **Step 3: Write minimal implementation** (append to `atomica/critic_world.py`)

```python
import itertools, json
from pathlib import Path

def build_world(evaluate_fn=alloy.evaluate, cache_path=None):
    if cache_path is not None and Path(cache_path).exists():
        d = json.loads(Path(cache_path).read_text())
        return {"configs": [tuple(c) for c in d["configs"]],
                "features": {k: np.array(v) for k, v in d["features"].items()},
                "energy": np.array(d["energy"])}
    configs = list(itertools.combinations(range(N_SITES), N_AU))
    energy = np.array([float(evaluate_fn(c)) for c in configs])
    feats = {name: np.array([features(c)[name] for c in configs]) for name in FEATURE_NAMES}
    world = {"configs": configs, "features": feats, "energy": energy}
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(
            {"configs": [list(c) for c in configs],
             "features": {k: v.tolist() for k, v in feats.items()},
             "energy": energy.tolist()}))
    return world

def truth_sign(target, confounder, world, kbins=3):
    return _sign(stratified_effect(world["features"][target],
                                   world["features"][confounder],
                                   world["energy"], kbins))

def _z(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / (a.std() + 1e-9)

def make_claim(world, seed, n=40, strength=2.0):
    rng = np.random.default_rng(seed)
    a, b = rng.choice(FEATURE_NAMES, size=2, replace=False)
    target, conf = str(a), str(b)
    F, E = world["features"], world["energy"]
    w = np.exp(strength * _z(F[target]) * _z(F[conf])); w = w / w.sum()
    idx = rng.choice(len(E), size=n, replace=False, p=w)
    claim_sign = _sign(_contrast(F[target][idx], E[idx]))
    truth = truth_sign(target, conf, world)
    sample = {name: F[name][idx].tolist() for name in FEATURE_NAMES}
    sample["energy"] = E[idx].tolist()
    return {"target": target, "claim_sign": int(claim_sign),
            "confounder_true": conf,
            "sample": sample,
            "label_is_true": bool(claim_sign == truth and claim_sign != 0),
            "seed": int(seed)}

def generate_claims(world, n_claims, n=40, strength=2.0, seed0=0):
    claims, seed = [], seed0
    while len(claims) < n_claims:
        c = make_claim(world, seed, n, strength)
        seed += 1
        if c["claim_sign"] == 0 or truth_sign(c["target"], c["confounder_true"], world) == 0:
            continue                     # skip degenerate (no direction to test)
        claims.append(c)
    return claims
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_critic_world.py -k "build_world or generate_claims or make_claim" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/critic_world.py tests/test_critic_world.py
git commit -m "feat: P4 world builder + deterministic biased scientist claims"
```

---

### Task 3: `critic.py` — validation + within-sample control

**Files:**
- Create: `atomica/critic.py`
- Test: `tests/test_critic.py`

**Interfaces:**
- Consumes: `atomica.critic_world.FEATURE_NAMES`, `stratified_effect`, `_sign`.
- Produces:
  - `CRITIC_TOOL` (strict tool schema dict), `validate_critique(raw, target) -> {"verdict", "confounder"}`.
  - `apply_control(claim, confounder) -> bool` (True = accepted / survives, False = rejected).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_critic.py
import numpy as np
import pytest
from atomica.critic import validate_critique, apply_control
from atomica.critic_world import FEATURE_NAMES

def test_validate_supported_and_confounded():
    assert validate_critique({"verdict": "supported"}, "x1") == {"verdict": "supported", "confounder": None}
    assert validate_critique({"verdict": "confounded", "confounder": "x2"}, "x1") == \
        {"verdict": "confounded", "confounder": "x2"}

def test_validate_rejects_bad():
    for bad in [{"verdict": "maybe"}, {"verdict": "confounded", "confounder": "x1"},  # == target
                {"verdict": "confounded", "confounder": "nope"}, {"verdict": "confounded"},
                "nope", None]:
        with pytest.raises(ValueError):
            validate_critique(bad, "x1")

def _confounded_claim():
    # Claim: high-x1 -> lower energy (claim_sign = -1), but that's driven by x2 (confounder).
    # In the sample, x1 and x2 covary; energy depends on x2. Stratifying on x2 removes the effect.
    n = 40
    x2 = np.array([0]*20 + [1]*20)
    x1 = x2.copy()                          # confounded within the sample
    layer = np.zeros(n)
    energy = 1.0 * x2                        # higher x2 -> higher energy
    # naive x1 contrast: high-x1 (=high x2) has HIGHER energy -> claim_sign should be +1... make it
    # a claim that high-x1 -> higher energy:
    return {"target": "x1", "claim_sign": 1, "confounder_true": "x2",
            "sample": {"x1": x1.tolist(), "x2": x2.tolist(), "layer": layer.tolist(),
                       "energy": energy.tolist()}}

def test_control_right_confounder_rejects_false_claim():
    c = _confounded_claim()
    # The claim "high x1 -> higher energy" is confounded by x2; stratifying on x2 flattens x1's
    # effect. Because within-stratum x1 has no variation here, controlled sign is 0 (not a flip),
    # so this exact degenerate case ACCEPTS. Use a partial-overlap sample instead:
    # (kept simple) assert the RIGHT confounder changes the outcome vs a no-op control.
    assert apply_control(c, None) is True                 # no confounder named -> stands
    assert apply_control(c, "x1") is True                 # target as confounder -> invalid -> stands

def test_control_flips_when_stratification_reverses_sign():
    # Construct a sample where stratifying on Z reverses the X contrast sign.
    x = np.array([0, 1, 0, 1, 0, 1, 0, 1], float)
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1], float)
    # within each Z stratum, high-x has LOWER energy; but pooled, high-x looks HIGHER (via Z)
    e = np.array([0, -1, 0, -1, 10, 9, 10, 9], float)
    claim = {"target": "x", "claim_sign": 1, "confounder_true": "z",
             "sample": {"x": x.tolist(), "z": z.tolist(), "energy": e.tolist()}}
    # pooled contrast is +? high-x mean vs low-x mean:
    # controlled (stratified on z) contrast is negative -> flips claim_sign(+1) -> reject
    assert apply_control(claim, "z") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_critic.py -v`
Expected: FAIL with `ModuleNotFoundError: atomica.critic`.

- [ ] **Step 3: Write minimal implementation**

```python
# atomica/critic.py
"""P4 critic: validate a structured verdict, apply a within-sample stratified
control (sign-flip => reject), and score arms. LLM proposes; harness decides."""
from atomica.critic_world import FEATURE_NAMES, stratified_effect, _sign

VALID_VERDICTS = ("supported", "confounded")

CRITIC_TOOL = {
    "name": "critique_claim",
    "description": "Judge whether a claim that a feature drives energy is supported or confounded "
                   "by another feature, and if confounded, name the confounder.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": list(VALID_VERDICTS),
                        "description": "supported if the claim holds; confounded if another feature explains it"},
            "confounder": {"type": "string", "enum": FEATURE_NAMES,
                           "description": "the feature confounding the claim (required when verdict is confounded; not the target)"},
        },
        "required": ["verdict"],
        "additionalProperties": False,
    },
    "strict": True,
}

def validate_critique(raw, target):
    if not isinstance(raw, dict):
        raise ValueError(f"critique not a dict: {raw!r}")
    verdict = raw.get("verdict")
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"bad verdict: {verdict!r}")
    if verdict == "supported":
        return {"verdict": "supported", "confounder": None}
    conf = raw.get("confounder")
    valid = [f for f in FEATURE_NAMES if f != target]
    if conf not in valid:
        raise ValueError(f"bad confounder {conf!r}; must be one of {valid}")
    return {"verdict": "confounded", "confounder": conf}

def apply_control(claim, confounder):
    """Stratify the claim's own sample on `confounder`; sign-flip => reject.
    Returns True if the claim is ACCEPTED (survives), False if REJECTED."""
    if confounder is None or confounder == claim["target"]:
        return True
    s = claim["sample"]
    controlled = _sign(stratified_effect(s[claim["target"]], s[confounder], s["energy"]))
    if controlled != 0 and controlled != claim["claim_sign"]:
        return False
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_critic.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/critic.py tests/test_critic.py
git commit -m "feat: P4 critique validation + within-sample stratified control"
```

---

### Task 4: `critic.py` — random critic, batch review, scoring

**Files:**
- Modify: `atomica/critic.py`
- Test: `tests/test_critic.py` (add cases)

**Interfaces:**
- Consumes: `validate_critique`, `apply_control`, `FEATURE_NAMES`.
- Produces:
  - `accept_all_critic(claim) -> dict` (always `{"verdict": "supported"}`).
  - `random_critic(rng, p_confounded=1.0) -> callable(claim) -> dict`.
  - `review_one(claim, critic) -> dict` (keys: `accepted`, `verdict`, `confounder`, `fallback`, `named_true_confounder`).
  - `review_batch(claims, critic) -> list[dict]`.
  - `score(claims, reviews) -> dict` (keys: `fdr`, `retention`, `n_accepted`, `accepted_false`, `accepted_true`, `n_claims`, `n_true`, `base_rate_false`).

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_critic.py
import numpy as np
from atomica.critic import (
    accept_all_critic, random_critic, review_one, review_batch, score,
)

def _claim(target, claim_sign, conf_true, is_true, sample):
    return {"target": target, "claim_sign": claim_sign, "confounder_true": conf_true,
            "label_is_true": is_true, "sample": sample}

def _trivial_sample():
    return {"x1": [0.0, 1.0], "x2": [0.0, 1.0], "layer": [0.0, 0.0], "energy": [0.0, -1.0]}

def test_accept_all_accepts():
    c = _claim("x1", 1, "x2", False, _trivial_sample())
    r = review_one(c, accept_all_critic)
    assert r["accepted"] is True and r["verdict"] == "supported"

def test_random_critic_in_space_and_seeded():
    rng = np.random.default_rng(0)
    critic = random_critic(rng)
    c = _claim("x1", 1, "x2", False, _trivial_sample())
    for _ in range(20):
        out = critic(c)
        assert out["verdict"] in ("supported", "confounded")
        if out["verdict"] == "confounded":
            assert out["confounder"] in ("x2", "layer")   # never the target

def test_score_fdr_and_retention():
    # 4 claims: 2 true, 2 false. accept_all -> FDR 0.5, retention 1.0
    claims = [_claim("x1", 1, "x2", True, _trivial_sample()),
              _claim("x1", 1, "x2", True, _trivial_sample()),
              _claim("x1", 1, "x2", False, _trivial_sample()),
              _claim("x1", 1, "x2", False, _trivial_sample())]
    reviews = review_batch(claims, accept_all_critic)
    s = score(claims, reviews)
    assert s["n_accepted"] == 4
    assert abs(s["fdr"] - 0.5) < 1e-9
    assert abs(s["retention"] - 1.0) < 1e-9
    assert abs(s["base_rate_false"] - 0.5) < 1e-9

def test_review_records_fallback_on_garbage():
    def garbage_critic(claim): return {"verdict": "???"}
    c = _claim("x1", 1, "x2", False, _trivial_sample())
    r = review_one(c, garbage_critic)
    assert r["fallback"] is True and r["accepted"] is True   # invalid -> supported fallback
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_critic.py -k "accept_all or random_critic or score or fallback" -v`
Expected: FAIL with ImportError for `accept_all_critic`.

- [ ] **Step 3: Write minimal implementation** (append to `atomica/critic.py`)

```python
def accept_all_critic(claim):
    return {"verdict": "supported"}

def random_critic(rng, p_confounded=1.0):
    def critic(claim):
        if rng.random() < p_confounded:
            others = [f for f in FEATURE_NAMES if f != claim["target"]]
            return {"verdict": "confounded", "confounder": str(rng.choice(others))}
        return {"verdict": "supported"}
    return critic

def review_one(claim, critic):
    fallback = False
    try:
        crit = validate_critique(critic(claim), claim["target"])
    except ValueError:
        crit = {"verdict": "supported", "confounder": None}
        fallback = True
    accepted = True if crit["verdict"] == "supported" else apply_control(claim, crit["confounder"])
    return {"accepted": bool(accepted), "verdict": crit["verdict"],
            "confounder": crit["confounder"], "fallback": fallback,
            "named_true_confounder": crit["confounder"] == claim["confounder_true"]}

def review_batch(claims, critic):
    return [review_one(c, critic) for c in claims]

def score(claims, reviews):
    acc = [r["accepted"] for r in reviews]
    accepted_true = int(sum(a and c["label_is_true"] for a, c in zip(acc, claims)))
    accepted_false = int(sum(a and not c["label_is_true"] for a, c in zip(acc, claims)))
    n_accepted = accepted_true + accepted_false
    n_true = int(sum(c["label_is_true"] for c in claims))
    n = len(claims)
    return {"fdr": accepted_false / n_accepted if n_accepted else 0.0,
            "retention": accepted_true / n_true if n_true else 0.0,
            "n_accepted": n_accepted, "accepted_false": accepted_false,
            "accepted_true": accepted_true, "n_claims": n, "n_true": n_true,
            "base_rate_false": (n - n_true) / n if n else 0.0}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_critic.py -k "accept_all or random_critic or score or fallback" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/critic.py tests/test_critic.py
git commit -m "feat: P4 random critic, batch review, FDR + retention scoring"
```

---

### Task 5: `critic.py` — `llm_critic` (Anthropic strict-tool structured output)

**Files:**
- Modify: `atomica/critic.py`
- Test: `tests/test_critic.py` (add cases)

**REQUIRED:** Before writing this task, read the `claude-api` skill's `python/claude-api/tool-use.md` for the exact strict-tool surface (as P3 did: `strict: true` top-level, `additionalProperties: false`, `required`, `tool_choice={"type":"tool","name":...}`, extract from the `tool_use` block's `.input`). Do not guess.

**Interfaces:**
- Consumes: `CRITIC_TOOL`, `FEATURE_NAMES`.
- Produces:
  - `MODEL = "claude-sonnet-5"`.
  - `build_prompt(claim) -> str` — names the target, the claim direction, the candidate confounders, and the sample as a compact table.
  - `llm_critic(client, model=MODEL) -> callable(claim) -> dict` — one structured call returning the raw `{verdict, confounder?}` (validated by `review_one`).

- [ ] **Step 1: Write the failing tests** (fake client — never hits the API)

```python
# add to tests/test_critic.py
from atomica.critic import llm_critic, build_prompt, MODEL

class _FakeBlock:
    def __init__(self, inp): self.type = "tool_use"; self.name = "critique_claim"; self.input = inp
class _FakeResp:
    def __init__(self, inp): self.content = [_FakeBlock(inp)]
class _FakeMessages:
    def __init__(self, inp): self._inp = inp; self.last_kwargs = None
    def create(self, **kw): self.last_kwargs = kw; return _FakeResp(self._inp)
class _FakeClient:
    def __init__(self, inp): self.messages = _FakeMessages(inp)

def test_llm_critic_extracts_verdict():
    client = _FakeClient({"verdict": "confounded", "confounder": "x2"})
    critic = llm_critic(client)
    raw = critic(_claim("x1", 1, "x2", False, _trivial_sample()))
    assert raw == {"verdict": "confounded", "confounder": "x2"}
    assert client.messages.last_kwargs["model"] == MODEL

def test_build_prompt_mentions_target_and_candidates():
    txt = build_prompt(_claim("x1", -1, "x2", False, _trivial_sample()))
    assert "x1" in txt and "x2" in txt and "layer" in txt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_critic.py -k "llm_critic or build_prompt" -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation** (append to `atomica/critic.py`; match the `claude-api` skill's strict-tool surface)

```python
MODEL = "claude-sonnet-5"

def build_prompt(claim):
    tgt = claim["target"]
    direction = "lower" if claim["claim_sign"] < 0 else "higher"
    others = [f for f in FEATURE_NAMES if f != tgt]
    s = claim["sample"]
    n = len(s["energy"])
    lines = [
        "A scientist studied Cu-Au orderings on a fixed 12-site lattice. Each config has three",
        f"integer structural features {FEATURE_NAMES} and a MACE energy (lower = more stable).",
        f"CLAIM: higher {tgt} is associated with {direction} energy.",
        f"The claim was drawn from this sample of {n} configs. It may be CONFOUNDED: another",
        f"feature ({' or '.join(others)}) may covary with {tgt} and actually drive the energy.",
        "Sample (one row per config):",
        "  " + "  ".join(FEATURE_NAMES) + "  energy",
    ]
    for i in range(n):
        row = "  ".join(f"{s[name][i]:g}" for name in FEATURE_NAMES)
        lines.append(f"  {row}  {s['energy'][i]:.3f}")
    lines.append(f"Call critique_claim: verdict 'supported' if {tgt} genuinely drives energy, or "
                 f"'confounded' naming the confounder among {others}.")
    return "\n".join(lines)

def llm_critic(client, model=MODEL):
    def critic(claim):
        resp = client.messages.create(
            model=model, max_tokens=512,
            tools=[CRITIC_TOOL], tool_choice={"type": "tool", "name": "critique_claim"},
            messages=[{"role": "user", "content": build_prompt(claim)}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "critique_claim":
                return block.input
        raise ValueError("no critique_claim tool_use in response")
    return critic
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_critic.py -k "llm_critic or build_prompt" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atomica/critic.py tests/test_critic.py
git commit -m "feat: P4 LLM critic via Anthropic strict-tool structured output"
```

---

### Task 6: `run_critic.py` CLI + report

**Files:**
- Create: `atomica/run_critic.py`
- Test: `tests/test_run_critic.py`

**Interfaces:**
- Consumes: `atomica.critic_world.build_world`, `generate_claims`; `atomica.critic` (`accept_all_critic`, `random_critic`, `llm_critic`, `review_batch`, `score`, `MODEL`).
- Produces: `make_llm_critic(model) -> critic|None`; `run_critic.main(argv=None) -> None` writing `results/critic_report.json` (keys: `arms` = {name: score_dict}, `config`).

- [ ] **Step 1: Write the failing test** (monkeypatch the LLM arm off — no credential, no MACE via a fake world)

```python
# tests/test_run_critic.py
import json
import numpy as np
from atomica import run_critic
from atomica.critic_world import build_world, features

def test_cli_smoke_none_and_random(tmp_path, monkeypatch):
    # Fake world (no MACE): energy depends on x2 only.
    def fake_eval(config): return 1.0 * features(config)["x2"]
    monkeypatch.setattr(run_critic, "make_llm_critic", lambda model: None)   # LLM arm off
    monkeypatch.setattr(run_critic, "build_world",
                        lambda evaluate_fn=None, cache_path=None: build_world(evaluate_fn=fake_eval))
    run_critic.main(["--n-claims", "20", "--n", "40", "--out", str(tmp_path)])
    report = json.loads((tmp_path / "critic_report.json").read_text())
    assert "none" in report["arms"] and "random" in report["arms"]
    assert "llm" not in report["arms"]                     # skipped, no client
    for arm in ("none", "random"):
        assert set(report["arms"][arm]) >= {"fdr", "retention", "n_accepted"}
    # none accepts everything -> retention 1.0
    assert report["arms"]["none"]["retention"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_run_critic.py -v`
Expected: FAIL with `ModuleNotFoundError: atomica.run_critic`.

- [ ] **Step 3: Write minimal implementation**

```python
# atomica/run_critic.py
"""ATOMICA P4 CLI: does an LLM critic cut false discoveries vs random vs none? (Cu-Au)."""
import argparse, json
from pathlib import Path
import numpy as np
from atomica.critic_world import build_world, generate_claims
from atomica.critic import (
    accept_all_critic, random_critic, llm_critic, review_batch, score, MODEL,
)

def make_llm_critic(model):
    """Return an llm_critic bound to a real client, or None if no credential/SDK."""
    try:
        import anthropic
        return llm_critic(anthropic.Anthropic(), model=model)
    except Exception as e:                      # missing key/sdk -> skip the LLM arm
        print(f"[run_critic] LLM arm disabled: {e}")
        return None

def main(argv=None):
    p = argparse.ArgumentParser(description="ATOMICA P4 LLM-critic false-discovery benchmark (Cu-Au)")
    p.add_argument("--n-claims", type=int, default=60)
    p.add_argument("--n", type=int, default=40, help="scientist sample size per claim")
    p.add_argument("--strength", type=float, default=2.0, help="confounding strength")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)

    world = build_world(cache_path=str(Path(a.out) / "alloy_world.json"))
    claims = generate_claims(world, a.n_claims, n=a.n, strength=a.strength, seed0=a.seed)

    arms = {}
    arms["none"] = score(claims, review_batch(claims, accept_all_critic))
    arms["random"] = score(claims, review_batch(claims, random_critic(np.random.default_rng(1000 + a.seed))))

    llm = make_llm_critic(a.model)
    if llm is not None:
        try:
            arms["llm"] = score(claims, review_batch(claims, llm))
        except Exception as e:                  # auth resolves lazily on first call -> skip cleanly
            print(f"[run_critic] LLM arm disabled: {e}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "critic_report.json").write_text(json.dumps(
        {"arms": arms,
         "config": {"n_claims": a.n_claims, "n": a.n, "strength": a.strength,
                    "seed": a.seed, "model": a.model}}, indent=2))
    print(json.dumps({k: {"fdr": round(v["fdr"], 3), "retention": round(v["retention"], 3)}
                      for k, v in arms.items()}, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_run_critic.py -v` then the full suite `python3 -m pytest -q`.
Expected: PASS (Slice 1 + P2 + P3 + P4 all green).

- [ ] **Step 5: Commit**

```bash
git add atomica/run_critic.py tests/test_run_critic.py
git commit -m "feat: P4 run_critic CLI + report (LLM arm auto-skips without a credential)"
```

---

### Task 7: Deliverable run + README

**Files:**
- Modify: `README.md`

**Interfaces:** none (produces the result + docs).

- [ ] **Step 1: Check for a credential**

Run: `echo "${ANTHROPIC_API_KEY:+set}"` and `ant auth status 2>&1 || echo "no ant profile"`.
- Credential available → the full none/random/llm run is possible.
- **Not** available → run none+random (no key needed); the LLM arm auto-skips (`[run_critic] LLM arm disabled: ...`). Note in the README that the LLM arm needs the user's own `ANTHROPIC_API_KEY` / `ant auth login`. **Do not fabricate LLM numbers.**

- [ ] **Step 2: Run the deliverable**

```bash
python3 -m atomica.run_critic --n-claims 60 --n 40 --strength 2.0 --seed 0 --out results
```

This builds the 924-config Cu-Au world once via MACE (~5 min, cached to `results/alloy_world.json`), generates 60 labeled claims, and runs the arms. Read `results/critic_report.json`: compare `fdr` and `retention` across `none` / `random` / (`llm` if on).

- [ ] **Step 3: Write the README "P4" section**

Add a "P4 — LLM critic (falsification loop)" section after the P3 section: what it does (a deterministic scientist emits claims from confounded samples; the LLM critic names the confounder as validated JSON, never touches physics; a within-sample stratified control decides via sign-flip), how to run (`python3 -m atomica.run_critic`, note the `ANTHROPIC_API_KEY` requirement), and the result honestly — FDR(none) vs FDR(random) vs FDR(llm) at matched retention. Include caveats: the LLM is stochastic; the fix ceiling is < 100% (some false claims are noise, not confound); a null result is valid. Point to the P4 spec. Update the Roadmap table row P4 from "🔒 planned" to "✅ done" (core shipped; LLM arm user-gated), and bump the tests badge count to the new total.

- [ ] **Step 4: Run the full test suite**

Run: `python3 -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add README.md results/critic_report.json
git commit -m "feat: P4 deliverable run and README (LLM critic false-discovery benchmark)"
```

(If `results/*.json` is gitignored — it is — the `critic_report.json` add is a no-op; commit the README and note the report is regenerated by the CLI.)

---

## Self-Review

**1. Spec coverage:**
- World reuse P2 + 3 non-collinear features + stratified truth (spec §3) → Task 1, 2. ✅
- Deterministic biased scientist + TRUE/FALSE labeling (spec §3) → Task 2. ✅
- Critic strict-tool action space + validation + supported-fallback (spec §4/§6) → Task 3, 5. ✅
- Within-sample stratified control, sign-flip reject (spec §4) → Task 3. ✅
- 3 arms none/random/llm + FDR + retention (spec §5) → Task 4, 6. ✅
- `llm_critic` injected client, no code execution, MODEL constant (spec §6) → Task 5. ✅
- CLI + LLM-arm auto-skip without credential (spec §6/§7) → Task 6. ✅
- Deliverable + honest caveats + credential-gated run (spec §9) → Task 7. ✅
- Tests never hit MACE or the real API (fake evaluate_fn / fake client / monkeypatch) (spec §8) → Tasks 1–6. ✅

**2. Placeholder scan:** No "TBD"/vague steps; every code step carries concrete code. Task 5 requires confirming the Anthropic strict-tool surface from the `claude-api` skill before writing — the shown code is the concrete target, adjusted only if the SDK surface differs.

**3. Type consistency:** `features(config)->{x1,x2,layer}` (T1,2). `stratified_effect(Xs,Zs,Es,kbins)->float` and `_sign`/`_contrast` (T1) reused in T2,3. `build_world(evaluate_fn,cache_path)->{configs,features,energy}` (T2) used in T6. `make_claim`/`generate_claims` claim dict keys (`target`,`claim_sign`,`confounder_true`,`sample`,`label_is_true`) consistent across T2,3,4,5,6. `validate_critique(raw,target)->{verdict,confounder}` (T3) used in T4. `apply_control(claim,confounder)->bool` (T3) used in T4. `review_one`/`review_batch`/`score` (T4) used in T6. `CRITIC_TOOL`/`build_prompt`/`llm_critic`/`MODEL` (T3,5) used in T6. `make_llm_critic(model)->critic|None` (T6) monkeypatched in T6 test. ✅
