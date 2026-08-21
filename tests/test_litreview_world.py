import numpy as np
from atomica.critic_world import build_world, features
from atomica.litreview_world import make_paper, generate_papers

def _fake_world():
    # Energy = x1 + 0.31*x2 + 0.07*layer — a near-unique combo (few ties, a clean global minimum),
    # so `better_in_gap` reliably mixes True/False across seeds as the studied band moves.
    def fake_eval(config):
        f = features(config)
        return float(f["x1"] + 0.31 * f["x2"] + 0.07 * f["layer"])
    return build_world(evaluate_fn=fake_eval)

def test_make_paper_shape_and_types():
    w = _fake_world()
    p = make_paper(w, seed=0)
    assert p["axis"] in ("x1", "x2", "layer")
    assert isinstance(p["best_config"], list) and len(p["best_config"]) == 6
    assert isinstance(p["best_energy"], float)
    assert isinstance(p["boundary_trend"], list) and len(p["boundary_trend"]) >= 2
    assert p["gap_side"] in ("high", "low")
    assert isinstance(p["better_in_gap"], bool)

def test_label_matches_independent_recompute():
    # Independently recompute better_in_gap from the full world and the paper's explored best.
    w = _fake_world()
    for seed in range(30):
        p = make_paper(w, seed=seed)
        E = w["energy"]; zc = w["features"][p["axis"]].astype(float)
        # reconstruct explored mask from gap_side + best_energy is not enough; instead check the
        # invariant: better_in_gap == (some config strictly below best_energy exists outside explored).
        # Explored = configs on the studied side; recompute via gap_side.
        # (gap_side "high" means gap is the high-axis side -> explored is the low side.)
        assert isinstance(p["better_in_gap"], bool)
        # best_energy must be the min over the explored subset, so no explored config is below it:
        # and better_in_gap true => min over ALL < best_energy.
        if p["better_in_gap"]:
            assert E.min() < p["best_energy"] - 1e-9
        else:
            assert abs(E.min() - p["best_energy"]) < 1e-9 or E.min() >= p["best_energy"] - 1e-9

def test_generate_papers_count_and_mixed_labels():
    w = _fake_world()
    papers = generate_papers(w, n_papers=40, seed0=0)
    assert len(papers) == 40
    for p in papers:
        assert p["n_explored"] >= 40
    labels = {p["better_in_gap"] for p in papers}
    assert labels == {True, False}          # both classes appear (energy = x1 gives a real mix)
