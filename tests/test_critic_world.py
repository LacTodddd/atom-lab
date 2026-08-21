import numpy as np
from atomica.critic_world import (
    features, stratified_effect, sign, _contrast, FEATURE_NAMES, N_SITES, N_AU,
    build_world, truth_sign, make_claim, generate_claims,
)

def test_features_shape_and_range():
    f = features((0, 1, 2, 3, 4, 5))
    assert set(f) == set(FEATURE_NAMES)
    assert all(isinstance(v, int) for v in f.values())
    assert 0 <= f["layer"] <= 4          # one (100) plane holds 4 sites

def test_sign_and_contrast():
    assert sign(-0.5) == -1 and sign(0.5) == 1 and sign(0.0) == 0
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
