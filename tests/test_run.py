from pathlib import Path
from atomica.run import main

def test_cli_smoke(tmp_path):
    out = tmp_path / "results"
    main(["--n", "2", "--budget", "6", "--seeds", "2", "--methods", "random",
          "--out", str(out)])
    assert (out / "random_N2_seed0.json").exists()
    assert (out / "random_N2_seed1.json").exists()
    assert list(out.glob("convergence_N2.png"))
