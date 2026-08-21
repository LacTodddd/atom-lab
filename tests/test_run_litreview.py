import json
from atomica import run_litreview
from atomica.critic_world import build_world, features

def test_cli_smoke_baseline_and_heuristic(tmp_path, monkeypatch):
    def fake_eval(config):                                        # near-unique combo, no MACE
        f = features(config); return float(f["x1"] + 0.31 * f["x2"] + 0.07 * f["layer"])
    monkeypatch.setattr(run_litreview, "make_llm_reviewer", lambda model: None)   # LLM arm off
    monkeypatch.setattr(run_litreview, "build_world",
                        lambda evaluate_fn=None, cache_path=None: build_world(evaluate_fn=fake_eval))
    run_litreview.main(["--n-papers", "20", "--seed", "0", "--out", str(tmp_path)])
    report = json.loads((tmp_path / "litreview_report.json").read_text())
    assert "baseline" in report["arms"] and "heuristic" in report["arms"]
    assert "llm" not in report["arms"]                    # skipped, no client
    for arm in ("baseline", "heuristic"):
        assert set(report["arms"][arm]) >= {"accuracy", "precision", "recall", "n"}
    assert report["arms"]["baseline"]["recall"] == 0.0    # always-False -> 0 recall
