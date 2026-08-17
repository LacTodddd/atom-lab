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
