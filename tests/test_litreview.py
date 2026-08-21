import pytest
from atomica.litreview import validate_prediction, baseline_reviewer, heuristic_reviewer

def _paper(trend, better=False):
    return {"axis": "x2", "region": "orderings with x2 in the low range", "n_explored": 100,
            "best_config": [0, 1, 2, 3, 4, 5], "best_energy": -44.0, "boundary_trend": trend,
            "gap_side": "high", "better_in_gap": better}

def test_validate_prediction():
    assert validate_prediction({"better_in_gap": True}) == {"better_in_gap": True}
    assert validate_prediction({"better_in_gap": False}) == {"better_in_gap": False}
    for bad in [{"better_in_gap": "yes"}, {"better_in_gap": 1}, {}, "nope", None]:
        with pytest.raises(ValueError):
            validate_prediction(bad)

def test_baseline_always_false():
    assert baseline_reviewer(_paper([-1, -2, -3, -4])) == {"better_in_gap": False}

def test_heuristic_reads_boundary_trend():
    # energy still improving toward the gap (last sub-band lowest) -> predict better_in_gap True
    assert heuristic_reviewer(_paper([-1.0, -2.0, -3.0, -4.0]))["better_in_gap"] is True
    # flat / worsening near the gap -> False
    assert heuristic_reviewer(_paper([-4.0, -4.0, -4.0, -4.0]))["better_in_gap"] is False
    assert heuristic_reviewer(_paper([-4.0, -3.0, -2.0, -1.0]))["better_in_gap"] is False
