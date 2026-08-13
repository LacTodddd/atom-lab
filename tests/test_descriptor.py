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
