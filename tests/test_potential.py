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
