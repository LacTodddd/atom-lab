PARAM_SPACE = {"k_acq": (0.0, 3.0), "pool": [40, 80, 160], "n_init": [5, 10, 20]}
DEFAULT_PARAMS = {"k_acq": 1.0, "pool": 100, "n_init": 10}

def validate_params(raw):
    if not isinstance(raw, dict):
        raise ValueError(f"params not a dict: {raw!r}")
    try:
        k = float(raw["k_acq"]); pool = int(raw["pool"]); n = int(raw["n_init"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"unparseable params: {raw!r}") from e
    lo, hi = PARAM_SPACE["k_acq"]
    k = min(max(k, lo), hi)
    pool = min(PARAM_SPACE["pool"], key=lambda x: abs(x - pool))
    n = min(PARAM_SPACE["n_init"], key=lambda x: abs(x - n))
    return {"k_acq": k, "pool": pool, "n_init": n}
