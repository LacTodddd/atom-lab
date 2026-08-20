import pytest
from atomica.strategist import validate_params, PARAM_SPACE, DEFAULT_PARAMS

def test_clamps_and_snaps():
    p = validate_params({"k_acq": 9.9, "pool": 70, "n_init": 100})
    assert p["k_acq"] == 3.0           # clamped to [0,3]
    assert p["pool"] == 80             # snapped to nearest of {40,80,160}
    assert p["n_init"] == 20           # snapped to nearest of {5,10,20}

def test_low_clamp():
    assert validate_params({"k_acq": -5, "pool": 40, "n_init": 5})["k_acq"] == 0.0

def test_rejects_unparseable():
    for bad in [{"k_acq": "x", "pool": 40, "n_init": 5}, {"pool": 40}, "nope", None]:
        with pytest.raises(ValueError):
            validate_params(bad)

def test_default_params_shape():
    assert DEFAULT_PARAMS == {"k_acq": 1.0, "pool": 100, "n_init": 10}
