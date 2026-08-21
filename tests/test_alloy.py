import numpy as np
from atomica.alloy import build_lattice, config_symbols, evaluate, sro_descriptor, brute_force_min

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

def test_sro_fixed_length_and_normalized():
    d = sro_descriptor((0, 1, 2, 3, 4, 5))
    assert d.shape == (3,)
    assert abs(d.sum() - 1.0) < 1e-9

def test_sro_symmetry_equivalent_configs_match():
    # Period shift is a lattice symmetry -> identical SRO.
    a = sro_descriptor((0, 1, 2, 3, 4, 5))
    b = sro_descriptor((4, 5, 6, 7, 8, 9))
    assert np.allclose(a, b)

def test_sro_all_au_pairs_only_auau():
    # If every neighbour bond is Au-Au (all 12 sites Au — not composition-valid, but a pure
    # descriptor check), Au-Cu and Cu-Cu bins are zero.
    d = sro_descriptor(tuple(range(12)))
    assert d[1] == 0.0 and d[2] == 0.0 and abs(d[0] - 1.0) < 1e-9

def test_brute_force_finds_min_on_toy():
    # Fake energy: lower when sites {0,1} are chosen. Space = C(4,2) = 6 configs.
    def fake(config, n_sites=4):
        return float(sum(config))  # minimized by (0,1)
    e, cfg, n = brute_force_min(fake, n_sites=4, n_au=2)
    assert n == 6
    assert cfg == (0, 1)
    assert e == 1.0

def test_relax_config_reports_a_physicality_check():
    from atomica.alloy import relax_config
    r = relax_config((0, 1, 4, 5, 8, 9), steps=5)
    assert set(r) == {"rigid_energy", "relaxed_energy", "energy_drop", "max_displacement"}
    assert r["relaxed_energy"] <= r["rigid_energy"] + 1e-9   # relaxation cannot raise the energy
    assert r["energy_drop"] >= -1e-9
    assert r["max_displacement"] >= 0.0

def test_brute_force_caches(tmp_path):
    calls = {"n": 0}
    def fake(config, n_sites=4):
        calls["n"] += 1
        return float(sum(config))
    p = tmp_path / "gt.json"
    brute_force_min(fake, n_sites=4, n_au=2, cache_path=p)
    first = calls["n"]
    brute_force_min(fake, n_sites=4, n_au=2, cache_path=p)  # served from cache
    assert calls["n"] == first
