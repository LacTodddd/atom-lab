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

def test_out_of_range_distances_clipped():
    # Cluster with one pair at distance 20 (exceeds r_max=8) plus normal distances
    pos = np.array([
        [0.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],  # pair distance 20 > r_max, will be clipped to 8
        [1.0, 0.0, 0.0],   # distance 1 from origin
        [0.0, 1.0, 0.0],   # distance 1 from origin
    ])
    h = distance_histogram(pos, bins=30, r_max=8.0)
    assert h.shape == (30,)
    assert abs(h.sum() - 1.0) < 1e-9, f"histogram sum {h.sum()} != 1.0"

def test_raises_on_single_atom():
    # Single atom has no pairwise distances; should raise ValueError
    pos = np.array([[0.0, 0.0, 0.0]])
    try:
        distance_histogram(pos)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "at least 2 atoms" in str(e)
