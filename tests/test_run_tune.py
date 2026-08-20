import json
from atomica import run_tune


def test_cli_smoke_random_only(tmp_path, monkeypatch):
    # Force the LLM path off so the smoke test needs no credential.
    monkeypatch.setattr(run_tune, "make_llm_proposer", lambda model: None)
    run_tune.main(["--rounds", "2", "--tune-seeds", "1", "--eval-seeds", "2",
                   "--budget", "12", "--trajectories", "1", "--out", str(tmp_path)])
    report = json.loads((tmp_path / "tune_report.json").read_text())
    assert "default" in report["comparison"]
    assert "random" in report["comparison"]
    assert "llm" not in report["comparison"]        # skipped, no client
