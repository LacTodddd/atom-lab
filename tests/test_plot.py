import json
from atomica.plot import success_rate, evals_to_target, write_metrics, KNOWN_MINIMA
from atomica.plot import make_figures  # ensure importable with new kwarg

def test_evals_to_target_hit_and_miss():
    hist = [[1, -1.0], [2, -3.0], [3, -3.0]]
    assert evals_to_target(hist, target=-3.0, tol=0.01) == 2
    assert evals_to_target(hist, target=-5.0, tol=0.01) is None

def test_success_rate_counts_seeds_reaching_target():
    good = [[1, -3.0]]
    bad = [[1, -1.0]]
    assert success_rate([good, bad, good], target=-3.0, tol=0.01) == 2 / 3


def _write_run(tmp_path, method, seed, n, history):
    path = tmp_path / f"{method}_N{n}_seed{seed}.json"
    path.write_text(json.dumps({
        "method": method, "n": n, "seed": seed, "budget": len(history),
        "history": history, "best_energy": history[-1][1],
        "best_positions": [[0.0, 0.0, 0.0]] * n,
    }))


def test_write_metrics_computes_success_rate(tmp_path):
    n = 13
    target = KNOWN_MINIMA[n]
    # "good" method: both seeds reach the target.
    _write_run(tmp_path, "good", 0, n, [[1, -10.0], [2, target]])
    _write_run(tmp_path, "good", 1, n, [[1, -10.0], [2, target - 0.001]])
    # "bad" method: neither seed reaches the target.
    _write_run(tmp_path, "bad", 0, n, [[1, -10.0], [2, -30.0]])
    _write_run(tmp_path, "bad", 1, n, [[1, -10.0], [2, -35.0]])

    paths = write_metrics(str(tmp_path), str(tmp_path))
    assert str(tmp_path / f"metrics_N{n}.json") in paths

    data = json.loads((tmp_path / f"metrics_N{n}.json").read_text())
    assert data["n"] == n
    assert data["known_min"] == target
    assert data["methods"]["good"]["success_rate"] == 1.0
    assert data["methods"]["good"]["evals_reached"] == 2
    assert data["methods"]["good"]["mean_evals_to_target"] == 2.0
    assert data["methods"]["bad"]["success_rate"] == 0.0
    assert data["methods"]["bad"]["evals_reached"] == 0
    assert data["methods"]["bad"]["mean_evals_to_target"] is None


def test_make_figures_accepts_known_minima_override(tmp_path):
    (tmp_path / "random_N12_seed0.json").write_text(json.dumps(
        {"method": "random", "n": 12, "seed": 0, "budget": 2,
         "history": [[1, -1.0], [2, -2.0]], "best_energy": -2.0, "best_config": [0,1,2,3,4,5]}))
    out = make_figures(results_dir=tmp_path, out_dir=tmp_path, known_minima={12: -2.5})
    assert out  # a PNG path was produced
