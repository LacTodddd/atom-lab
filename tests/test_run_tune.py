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


def test_cli_llm_arm_auth_failure_auto_skips(tmp_path, monkeypatch):
    # Simulate anthropic 0.125.0's real no-credential failure: Anthropic() construction
    # succeeds, but the first request raises a bare TypeError from header validation.
    def fake_proposer(history):
        raise TypeError(
            '"Could not resolve authentication method. Expected one of api_key, '
            'auth_token, or credentials to be set. Or for one of the `X-Api-Key` '
            'or `Authorization` headers to be explicitly omitted"'
        )
    monkeypatch.setattr(run_tune, "make_llm_proposer", lambda model: fake_proposer)
    run_tune.main(["--rounds", "2", "--tune-seeds", "1", "--eval-seeds", "2",
                   "--budget", "12", "--trajectories", "1", "--out", str(tmp_path)])
    report = json.loads((tmp_path / "tune_report.json").read_text())
    assert "random" in report["comparison"]
    assert "default" in report["comparison"]
    assert "llm" not in report["comparison"]        # auth failure auto-skips, doesn't crash
