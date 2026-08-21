import pytest
from atomica.litreview import validate_prediction, baseline_reviewer, heuristic_reviewer, review_one, review_batch, score
from atomica.litreview import llm_reviewer, build_prompt, MODEL

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

def test_review_one_strips_label_and_scores():
    seen = {}
    def spy_reviewer(view):
        seen.update(view)
        return {"better_in_gap": True}
    p = _paper([-1, -2, -3, -4], better=True)
    r = review_one(p, spy_reviewer)
    assert "better_in_gap" not in seen        # cheat-proof: label never handed to the reviewer
    assert r["predicted"] is True and r["correct"] is True and r["fallback"] is False

def test_review_one_fallback_on_garbage():
    def garbage(view): return {"better_in_gap": "nope"}
    p = _paper([-1, -2, -3, -4], better=False)
    r = review_one(p, garbage)
    assert r["fallback"] is True and r["predicted"] is False and r["correct"] is True

def test_score_accuracy_precision_recall():
    # 4 papers: labels [T, T, F, F]. baseline predicts all False -> acc .5, precision 0, recall 0.
    papers = [_paper([-1], True), _paper([-1], True), _paper([-1], False), _paper([-1], False)]
    reviews = review_batch(papers, baseline_reviewer)
    s = score(papers, reviews)
    assert abs(s["accuracy"] - 0.5) < 1e-9
    assert s["precision"] == 0.0 and s["recall"] == 0.0
    assert abs(s["base_rate_better"] - 0.5) < 1e-9
    assert s["n"] == 4

class _FakeBlock:
    def __init__(self, inp): self.type = "tool_use"; self.name = "predict_gap"; self.input = inp
class _FakeResp:
    def __init__(self, inp): self.content = [_FakeBlock(inp)]
class _FakeMessages:
    def __init__(self, inp): self._inp = inp; self.last_kwargs = None
    def create(self, **kw): self.last_kwargs = kw; return _FakeResp(self._inp)
class _FakeClient:
    def __init__(self, inp): self.messages = _FakeMessages(inp)

def test_llm_reviewer_extracts_prediction():
    client = _FakeClient({"better_in_gap": True})
    reviewer = llm_reviewer(client)
    raw = reviewer({"axis": "x2", "region": "orderings with x2 in the low range", "n_explored": 100,
                    "best_config": [0, 1, 2, 3, 4, 5], "best_energy": -44.0,
                    "boundary_trend": [-1.0, -2.0, -3.0, -4.0], "gap_side": "high"})
    assert raw == {"better_in_gap": True}
    assert client.messages.last_kwargs["model"] == MODEL

def test_build_prompt_mentions_region_and_trend_not_label():
    paper = {"axis": "x2", "region": "orderings with x2 in the low range", "n_explored": 100,
             "best_config": [0, 1, 2, 3, 4, 5], "best_energy": -44.0,
             "boundary_trend": [-1.0, -2.0, -3.0, -4.0], "gap_side": "high", "better_in_gap": True}
    txt = build_prompt(paper)
    assert "x2" in txt and "-44" in txt and "boundary" in txt.lower()
    # cheat-proof: prompt is independent of the hidden label
    paper2 = dict(paper); paper2["better_in_gap"] = False
    assert build_prompt(paper) == build_prompt(paper2)
