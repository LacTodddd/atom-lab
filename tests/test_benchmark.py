import json
from atomica.benchmark import run_benchmark, METHODS

def test_run_benchmark_writes_expected_files(tmp_path):
    paths = run_benchmark(n_values=[2], seeds=[0], budget=6,
                          methods={"random": METHODS["random"]}, out_dir=tmp_path)
    assert len(paths) == 1
    data = json.loads(open(paths[0]).read())
    assert data["method"] == "random"
    assert data["n"] == 2 and data["seed"] == 0 and data["budget"] == 6
    assert len(data["history"]) == 6
    assert data["best_energy"] < -0.9
    assert len(data["best_positions"]) == 2
