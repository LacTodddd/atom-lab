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
