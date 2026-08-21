import json
import numpy as np
from atomica import run_critic
from atomica.critic_world import build_world, features

def test_cli_smoke_none_and_random(tmp_path, monkeypatch):
    # Fake world (no MACE): energy depends on x2 only.
    def fake_eval(config): return 1.0 * features(config)["x2"]
    monkeypatch.setattr(run_critic, "make_llm_critic", lambda model: None)   # LLM arm off
    monkeypatch.setattr(run_critic, "build_world",
                        lambda evaluate_fn=None, cache_path=None: build_world(evaluate_fn=fake_eval))
    run_critic.main(["--n-claims", "20", "--n", "40", "--out", str(tmp_path)])
    report = json.loads((tmp_path / "critic_report.json").read_text())
    assert "none" in report["arms"] and "random" in report["arms"]
    assert "llm" not in report["arms"]                     # skipped, no client
    for arm in ("none", "random"):
        assert set(report["arms"][arm]) >= {"fdr", "retention", "n_accepted"}
    # none accepts everything -> retention 1.0
    assert report["arms"]["none"]["retention"] == 1.0
