# ATOMICA — P4: LLM Critic (Falsification Loop)

- **Date:** 2026-08-21
- **Status:** Approved (design), pending implementation plan
- **Builds on:** P2 (`atomica/alloy.py` — the Cu-Au FCC world, `evaluate`, `build_lattice`, the
  brute-force ground truth) and P3's LLM conventions (strict-tool structured output, injected client,
  credential-gated auto-skip).
- **Roadmap position:** Phase P4 — an LLM *critic* that tries to falsify a claim by proposing a
  control, the second time an LLM enters the loop.
- **Spike:** the world and estimators below were validated by a throwaway spike before this spec (see
  §9). All quoted rates (~27% base-rate, ~77% right-confounder fix, ~5% wrong-confounder fix) are from
  that spike on the real MACE world.

---

## 1. Context

P3 let an LLM *propose* hyperparameters. P4 gives the LLM the opposite, harder job from the vision
(§11 of the project plan): **criticism / falsification** — take a claim a naive "scientist" asserts
and try to *overturn* it by identifying a confounder and running a control. The vision's own example
is "the result may be confounded by composition; the comparison set is insufficient; run a controlled
experiment varying only the variable of interest."

The measured, cheat-proof question (project plan line 1038):

> Does an AI critic reduce false scientific conclusions?

As in every prior slice, the LLM never touches the physics. A deterministic scientist makes claims;
the LLM only names a confounder; the harness runs the statistical control and the ground truth decides.

## 2. The measurable claim

The world is the **P2 Cu-Au 12-site FCC lattice** (6 Au / 6 Cu, C(12,6)=924 orderings, each with an
exact MACE energy — already brute-forceable). A deterministic scientist generates a batch of **claims**
of the form *"feature X is associated with lower/higher energy."* Some claims are **TRUE** and some are
**FALSE** (confounded), decided against the exact controlled ground truth over all 924.

Three **critic arms** review each claim under identical machinery:

- **none** — accept every claim (no criticism). False-discovery rate = the batch's base rate (~27%).
- **random** — a critic that names a *random* confounder; the harness runs the control anyway.
- **llm** — the LLM critic reads the claim + the scientist's sample and names the confounder it
  believes drives the association.

> **Headline:** among **accepted** claims, does the LLM critic's **false-discovery rate (FDR)** fall
> below *none* and below *random*, without sacrificing **true-claim retention**? (LLM ≈ random ⇒ the
> LLM adds nothing — a valid, honestly-reported negative, as in Slice 1 / P3.)

The spike shows the ceiling is real and confounder-identification is the crux: stratifying on the
**right** confounder fixes ~77% of false claims (keeps 100% of true ones); the **wrong** confounder
fixes ~5%. So the arms genuinely separate.

## 3. The world, the scientist, and the ground truth

All of P4's statistics operate on a **table**: for each of the 924 configs, three features and one
energy. The table is built once from `alloy.evaluate` (cached MACE singleton) and cached to
`results/alloy_world.json`. **No P2/P3 file is modified** — the world module reuses `alloy` functions.

**Features (bounded, fixed set of 3, spike-validated non-collinear):**

| Name | Meaning |
|------|---------|
| `x1` | first-NN Au-Au pair count |
| `x2` | second-NN Au-Au pair count (corr with `x1` ≈ −0.43 → a real, non-collinear confounder) |
| `layer` | number of Au atoms in one (100) plane (≈ uncorrelated with `x1`/`x2`) |

**Truth labeler (Q1 = stratified effect):** the controlled effect of feature `X` holding confounder
`Z` fixed is the **Z-stratified mean-energy contrast** of high-`X` vs low-`X`, computed over all 924
(quantile-binned `Z`, median split on `X` within each bin, count-weighted average). Its **sign** is the
controlled ground truth. Stratification (not OLS) is chosen so truth and the critic's control use the
*same estimator* at different sample sizes.

**Scientist (deterministic, no LLM):** for a seed, pick a target feature `X` and a true confounder
`Z`, then draw a **biased sample** `S` of `n=40` configs in which `X` and `Z` are pushed to covary
(exponential weighting `exp(strength · x̂·ẑ)`, `strength≈2`). The claim's direction = the **naive**
high-`X`-vs-low-`X` energy contrast *within S* (ignoring `Z`). Because `S` is confounded, this naive
sign sometimes contradicts the stratified truth → a **FALSE** claim; otherwise **TRUE**. Over the batch
this yields a ~27% false base rate with no population-level rigging (the confound lives in the
scientist's biased comparison set — exactly the vision's "insufficient comparison set").

## 4. The critic and the control

**What the critic sees:** the claim (target feature `X` and direction) and the scientist's sample `S`
as a small table (each config's three feature values + its energy). It does **not** see the full 924 or
the truth label.

**Critic action space (strict structured output, P3-style):** a single strict tool `critique_claim`
returning `{verdict: "supported" | "confounded", confounder: "x1"|"x2"|"layer"}` (the `confounder`
field is required only when `verdict = "confounded"` and must be one of the two non-target features).
Every output is validated by the harness; an unparseable/invalid output falls back to `supported`
(recorded as `fallback: true`) — a critic that fails to produce a valid objection simply doesn't object.

**The control (within-sample, no extra budget):** when the critic says `confounded` and names `Z`, the
harness **stratifies the scientist's own sample `S` on `Z`** and recomputes the `X` contrast (same
stratified estimator as the truth). This is a pure statistical control — no fresh world draws, no
budget knob. **Reject rule (Q5):** if the controlled sign **flips** relative to the claim's direction,
the claim is **rejected**; otherwise **accepted**. A `supported` verdict accepts the claim with no
control run.

Because the confound is in-sample, naming the **right** `Z` reveals the truth (~77% of false claims
flip) and the **wrong** `Z` does not (~5%) — so the LLM's confounder choice is the whole game, and the
random arm (which names `Z` by chance) is a meaningful baseline.

## 5. Arms, metrics, and fairness

For a batch of `N=60` claims (mixed TRUE/FALSE, labeled by §3), each arm produces an **accepted set**:

- **FDR** = (false claims accepted) / (all accepted) — **primary**, lower is better.
- **True-claim retention** = (true claims accepted) / (all true claims) — **guard**, higher is better;
  stops a "reject everything" critic from gaming FDR to 0.
- (Reported alongside: base rate, per-arm accept count, and — for random/llm — how often the named
  confounder was the true one.)

**Fairness invariant:** all three arms share the identical control machinery (§4) and the identical
claim batch; only the *verdict + confounder choice* differs. The critic never sees the truth label or
the full world. This is P3's budget-fairness discipline restated for the critic.

## 6. Safety & the LLM boundary (inherited from P3)

- The LLM's entire action space is **one verdict + one categorical confounder**, returned as validated
  strict-tool JSON. It cannot emit or run code, choose the physics, or see the ground truth.
- Every output passes harness validation before use; invalid → `supported` fallback (`fallback: true`).
- **Credentials are the user's:** the SDK reads `ANTHROPIC_API_KEY` or an `ant auth login` profile; no
  key in code. If no credential is configured, the **llm arm auto-skips** (as P3's `run_tune` does) and
  the none/random arms still run. **Tests never hit the real API** — a fake/injected client.
- **Model:** default `claude-sonnet-5` (module constant `MODEL`, switchable via `--model`).
- Reproducibility via `numpy.random.default_rng(seed)` for scientist and random arm; the LLM arm is
  inherently stochastic (reported as a caveat).

## 7. Architecture

Two new modules (clean world/critic seam) plus a CLI. Reuses `alloy` unchanged.

- `atomica/critic_world.py` — the world and its statistics, **decoupled from MACE**:
  - `build_world(evaluate_fn=alloy.evaluate, cache_path=...) -> (configs, features, energies)` —
    enumerates C(12,6), evaluates, caches. Tests inject a fake `evaluate_fn` or a prebuilt small table;
    only the deliverable builds the real 924.
  - `features(config) -> {x1, x2, layer}`; `stratified_effect(X, Z, E) -> float` (the shared
    estimator); `truth_sign(X, Z, world)`.
  - `make_claim(world, seed, n=40, strength=2.0) -> Claim` — the deterministic biased scientist;
    `Claim` carries the target feature, direction, the sample `S` (indices + feature/energy values),
    the true confounder, and the TRUE/FALSE label (label used only for scoring, never shown to critics).
- `atomica/critic.py` — the critics and scoring:
  - `CRITIC_TOOL` schema + `validate_critique(raw) -> {verdict, confounder|None}`.
  - `apply_control(claim, confounder) -> accepted: bool` — stratify `S` on `confounder`, sign-flip rule.
  - `random_critic(rng)`, `llm_critic(client, model=MODEL)` (strict tool, injected client),
    `build_prompt(claim)`.
  - `review_batch(claims, critic) -> accepts`; `score(claims, accepts) -> {fdr, retention, ...}`;
    `MODEL`.
- `atomica/run_critic.py` — CLI: build/load world, generate `N` claims, run none/random/llm arms,
  write `results/critic_report.json` + a small table. `make_llm_critic(model)` returns `None` (arm
  auto-skips) if no credential/SDK, mirroring P3's `run_tune`.

### Data flow

```
build_world (MACE, cached) ─▶ make_claim × N (deterministic, labeled)
        claims ──▶ arm none   : accept all
              ──▶ arm random : random_critic  ─┐
              ──▶ arm llm    : llm_critic     ─┴▶ apply_control (stratify S on Z, sign-flip) ─▶ accepts
        accepts per arm ──▶ score (FDR, retention) ──▶ results/critic_report.json + table
```

## 8. Testing & validation

Tests must be **fast** and **never** hit MACE or the Anthropic API:

- **World stats on a synthetic table:** feed `build_world` a fake `evaluate_fn` (or construct a small
  table directly) with a *known* controlled relationship; assert `stratified_effect` / `truth_sign`
  recover it, and that `make_claim` produces both TRUE and FALSE claims across seeds.
- **Control (key):** on a hand-built confounded sample, `apply_control` with the *right* confounder
  flips a false claim (rejects) and with the *wrong* confounder does not; a true claim is retained
  either way.
- **Validation:** `validate_critique` accepts well-formed output, snaps/rejects a bad `confounder`, and
  falls back to `supported` on garbage.
- **Random critic** yields an in-space confounder; seeded → reproducible.
- **LLM critic** with a **fake client** returns the injected `tool_use` verdict; `build_prompt` mentions
  the features and the sample. No network, no key.
- **CLI smoke** (monkeypatch `make_llm_critic → None`): runs none+random, writes a report with `fdr` and
  `retention` for both arms and no `llm` arm.
- The real `llm_critic` runs only in the credential-gated deliverable.

## 9. Deliverable / honest caveats

Deliverable: `results/critic_report.json` + a README "P4" section answering — on `N` claims — does the
LLM critic's **FDR** fall below **random** and **none**, at comparable **retention**?

Honest caveats to report as-is:
- The LLM is **stochastic**; run one batch and report it as one sample (optionally a couple of seeds).
- The **fix ceiling is ~77%, not 100%** — some false claims are small-sample noise, not confound, and no
  confounder can fix those. The measured question is specifically **LLM vs random confounder-naming**.
- The world is a **small 12-site lattice** with three features and an injected (not natural
  population-level) confound; it is a controlled benchmark, not a claim about real alloy discovery.
- A null result (LLM ≈ random) is a result.

## 10. Dependencies & scope

- No new dependency (`anthropic` already added in P3). Credential is the user's; never committed.
- **In scope:** the world module (features, stratified truth, biased scientist, claim labeling), the
  critic module (validation, within-sample control, random + LLM critics, batch scoring), the CLI, tests
  with a synthetic world + fake client, one credential-gated deliverable run.
- **Out of scope (later phases):** the LLM as scientist; multi-round scientist⇄critic dialogue; controls
  that draw fresh configs under a budget; population-level (Simpson's) confounds; literature ingestion
  (P5); any world beyond the P2 Cu-Au lattice.

## 11. Open items to resolve during implementation

- Confirm the exact Anthropic strict-tool surface from the `claude-api` skill before writing
  `llm_critic` (as P3 did — `strict: true` top-level, `additionalProperties: false`, `required`).
- Pick the final `strength` and quantile-bin count so the base rate lands ~25–35% and the right/wrong
  confounder gap stays wide (spike used `strength≈2`, 3 bins).
- Decide whether the deliverable runs one batch or a few seeds for the LLM arm (cost vs. stability).
