from atomica import run_alloy

def test_cli_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(run_alloy, "evaluate", lambda config, n_sites=12: float(sum(config)))
    run_alloy.main(["--budget", "6", "--seeds", "2", "--methods", "random",
                    "--out", str(tmp_path)])
    assert (tmp_path / "alloy_ground_truth.json").exists()
    assert (tmp_path / "random_N12_seed0.json").exists()
    assert list(tmp_path.glob("convergence_N12.png"))
