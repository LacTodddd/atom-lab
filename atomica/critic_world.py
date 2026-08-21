"""P4 critic world: Cu-Au features, stratified controlled-effect estimator,
and a deterministic biased 'scientist'. Reuses P2's alloy module unchanged."""
import numpy as np
from atomica import alloy

N_SITES, N_AU = 12, 6
FEATURE_NAMES = ["x1", "x2", "layer"]

_GEOM = None
def _geometry():
    global _GEOM
    if _GEOM is None:
        at = alloy.build_lattice(N_SITES)
        D = at.get_all_distances(mic=True)
        dv = np.unique(np.round(D[D > 1e-6], 3))
        s1, s2 = float(dv[0]), float(dv[1])
        zc = np.round(at.get_positions()[:, 2], 2)
        layer0 = list(np.where(zc == sorted(np.unique(zc))[0])[0])
        _GEOM = (D, s1, s2, layer0)
    return _GEOM

def features(config):
    D, s1, s2, layer0 = _geometry()
    au = np.zeros(N_SITES, bool); au[list(config)] = True
    def pairs(shell):
        return int(sum(au[i] and au[j] and abs(D[i, j] - shell) < 1e-2
                       for i in range(N_SITES) for j in range(i + 1, N_SITES)))
    return {"x1": pairs(s1), "x2": pairs(s2), "layer": int(au[layer0].sum())}

def sign(v, tol=1e-9):
    return 0 if abs(v) < tol else (1 if v > 0 else -1)

def _contrast(Xs, Es):
    """energy(high-X) - energy(low-X), median split on X."""
    Xs, Es = np.asarray(Xs, float), np.asarray(Es, float)
    m = np.median(Xs)
    hi, lo = Es[Xs > m], Es[Xs <= m]
    if len(hi) == 0 or len(lo) == 0:
        return 0.0
    return float(hi.mean() - lo.mean())

def stratified_effect(Xs, Zs, Es, kbins=3):
    """Z-stratified X-contrast, count-weighted over bins."""
    Xs, Zs, Es = np.asarray(Xs, float), np.asarray(Zs, float), np.asarray(Es, float)
    diffs, wts = [], []
    qs = np.quantile(Zs, np.linspace(0, 1, kbins + 1))
    for b in range(kbins):
        upper = (Zs <= qs[b + 1]) if b == kbins - 1 else (Zs < qs[b + 1])
        m = (Zs >= qs[b]) & upper
        if m.sum() < 3:
            continue
        d = _contrast(Xs[m], Es[m])
        if d != 0:
            diffs.append(d); wts.append(int(m.sum()))
    if not diffs:
        return 0.0
    return float(np.average(diffs, weights=wts))

import itertools, json
from pathlib import Path

def build_world(evaluate_fn=alloy.evaluate, cache_path=None):
    if cache_path is not None and Path(cache_path).exists():
        d = json.loads(Path(cache_path).read_text())
        return {"configs": [tuple(c) for c in d["configs"]],
                "features": {k: np.array(v) for k, v in d["features"].items()},
                "energy": np.array(d["energy"])}
    configs = list(itertools.combinations(range(N_SITES), N_AU))
    energy = np.array([float(evaluate_fn(c)) for c in configs])
    feats = {name: np.array([features(c)[name] for c in configs]) for name in FEATURE_NAMES}
    world = {"configs": configs, "features": feats, "energy": energy}
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(
            {"configs": [list(c) for c in configs],
             "features": {k: v.tolist() for k, v in feats.items()},
             "energy": energy.tolist()}))
    return world

def truth_sign(target, confounder, world, kbins=3):
    return sign(stratified_effect(world["features"][target],
                                   world["features"][confounder],
                                   world["energy"], kbins))

def _z(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / (a.std() + 1e-9)

def make_claim(world, seed, n=40, strength=2.0):
    rng = np.random.default_rng(seed)
    a, b = rng.choice(FEATURE_NAMES, size=2, replace=False)
    target, conf = str(a), str(b)
    F, E = world["features"], world["energy"]
    w = np.exp(strength * _z(F[target]) * _z(F[conf])); w = w / w.sum()
    idx = rng.choice(len(E), size=n, replace=False, p=w)
    claim_sign = sign(_contrast(F[target][idx], E[idx]))
    truth = truth_sign(target, conf, world)
    sample = {name: F[name][idx].tolist() for name in FEATURE_NAMES}
    sample["energy"] = E[idx].tolist()
    return {"target": target, "claim_sign": int(claim_sign),
            "confounder_true": conf,
            "sample": sample,
            "label_is_true": bool(claim_sign == truth and claim_sign != 0),
            "seed": int(seed)}

def generate_claims(world, n_claims, n=40, strength=2.0, seed0=0):
    claims, seed = [], seed0
    while len(claims) < n_claims:
        c = make_claim(world, seed, n, strength)
        seed += 1
        if c["claim_sign"] == 0 or truth_sign(c["target"], c["confounder_true"], world) == 0:
            continue                     # skip degenerate (no direction to test)
        claims.append(c)
    return claims
