import numpy as np
from atomica.alloy_search import random_config, mutate_swap, random_search, crossover, genetic_search, active_learning_search

def _fake(config, n_sites=12):
    return float(sum(config))  # deterministic, minimized by choosing the lowest indices

def test_random_config_composition():
    rng = np.random.default_rng(0)
    for _ in range(50):
        c = random_config(12, 6, rng)
        assert len(set(c)) == 6 and all(0 <= i < 12 for i in c)

def test_mutate_preserves_composition():
    rng = np.random.default_rng(0)
    c = (0, 1, 2, 3, 4, 5)
    for _ in range(50):
        c = mutate_swap(c, 12, rng)
        assert len(set(c)) == 6

def test_random_search_history_valid():
    hist, best = random_search(_fake, 12, 6, budget=20, seed=0)
    assert len(hist) == 20
    assert [h[0] for h in hist] == list(range(1, 21))
    energies = [h[1] for h in hist]
    assert all(energies[i] >= energies[i + 1] - 1e-9 for i in range(len(energies) - 1))
    assert len(set(best)) == 6

def test_crossover_preserves_composition():
    rng = np.random.default_rng(0)
    p1, p2 = (0, 1, 2, 3, 4, 5), (2, 3, 6, 7, 8, 9)
    for _ in range(50):
        child = crossover(p1, p2, 12, 6, rng)
        assert len(set(child)) == 6

def test_genetic_search_history_valid():
    hist, best = genetic_search(_fake, 12, 6, budget=25, seed=1)
    assert len(hist) == 25
    energies = [h[1] for h in hist]
    assert all(energies[i] >= energies[i + 1] - 1e-9 for i in range(len(energies) - 1))
    assert len(set(best)) == 6

def test_active_learning_history_valid():
    hist, best = active_learning_search(_fake, 12, 6, budget=20, seed=2)
    assert len(hist) == 20
    energies = [h[1] for h in hist]
    assert all(energies[i] >= energies[i + 1] - 1e-9 for i in range(len(energies) - 1))
    assert len(set(best)) == 6

def test_active_learning_calls_evaluate_exactly_budget():
    calls = {"n": 0}
    def counting(config, n_sites=12):
        calls["n"] += 1
        return _fake(config)
    active_learning_search(counting, 12, 6, budget=18, seed=0)
    assert calls["n"] == 18  # surrogate predictions cost no budget
