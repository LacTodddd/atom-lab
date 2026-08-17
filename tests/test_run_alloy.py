from atomica import run_alloy

def test_cli_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(run_alloy, "evaluate", lambda config, n_sites=12: float(sum(config)))
    run_alloy.main(["--budget", "6", "--seeds", "2", "--methods", "random",
                    "--out", str(tmp_path)])
    assert (tmp_path / "alloy_ground_truth.json").exists()
    assert (tmp_path / "random_N12_seed0.json").exists()
    assert list(tmp_path.glob("convergence_N12.png"))

def test_run_alloy_merges_known_minima(tmp_path, monkeypatch):
    monkeypatch.setattr(run_alloy, "evaluate", lambda config, n_sites=12: float(sum(config)))
    captured = {}
    def fake_make_figures(results_dir="results", out_dir="results", known_minima=None,
                           title_fmt="LJ-{n}", xlabel="relaxations"):
        captured["km"] = known_minima
        captured["title_fmt"] = title_fmt
        captured["xlabel"] = xlabel
        return []
    monkeypatch.setattr(run_alloy, "make_figures", fake_make_figures)
    run_alloy.main(["--budget", "6", "--seeds", "1", "--methods", "random", "--out", str(tmp_path)])
    assert 12 in captured["km"] and 13 in captured["km"] and 38 in captured["km"]
    assert captured["title_fmt"] == "Cu-Au {n}-site ordering"
    assert captured["xlabel"] == "MACE evaluations"
