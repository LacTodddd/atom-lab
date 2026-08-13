from atomica.plot import success_rate, evals_to_target

def test_evals_to_target_hit_and_miss():
    hist = [[1, -1.0], [2, -3.0], [3, -3.0]]
    assert evals_to_target(hist, target=-3.0, tol=0.01) == 2
    assert evals_to_target(hist, target=-5.0, tol=0.01) is None

def test_success_rate_counts_seeds_reaching_target():
    good = [[1, -3.0]]
    bad = [[1, -1.0]]
    assert success_rate([good, bad, good], target=-3.0, tol=0.01) == 2 / 3
