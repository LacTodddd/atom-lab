import json
from atomica.benchmark import run_alloy_benchmark
from atomica.alloy_search import random_search

def _fake(config, n_sites=12):
    return float(sum(config))

def test_run_alloy_benchmark_writes_json(tmp_path):
    paths = run_alloy_benchmark({"random": random_search}, seeds=[0, 1], budget=6,
                                evaluate=_fake, n_sites=12, n_au=6, out_dir=tmp_path)
    assert len(paths) == 2
    d = json.loads(open(paths[0]).read())
    assert d["method"] == "random" and d["budget"] == 6 and d["n"] == 12
    assert len(d["history"]) == 6
    assert len(d["best_config"]) == 6
