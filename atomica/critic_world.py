"""P4 critic world: Cu-Au features, stratified controlled-effect estimator,
and a deterministic biased 'scientist'. Reuses P2's alloy module unchanged."""
import numpy as np
from atomica import alloy

N_SITES, N_AU = 12, 6
FEATURE_NAMES = ["x1", "x2", "layer"]

def _geometry():
    at = alloy.build_lattice(N_SITES)
    D = at.get_all_distances(mic=True)
    dv = np.unique(np.round(D[D > 1e-6], 3))
    s1, s2 = float(dv[0]), float(dv[1])
    zc = np.round(at.get_positions()[:, 2], 2)
    layer0 = list(np.where(zc == sorted(np.unique(zc))[0])[0])
    return D, s1, s2, layer0

_D, _S1, _S2, _LAYER0 = _geometry()

def features(config):
    au = np.zeros(N_SITES, bool); au[list(config)] = True
    def pairs(shell):
        return int(sum(au[i] and au[j] and abs(_D[i, j] - shell) < 1e-2
                       for i in range(N_SITES) for j in range(i + 1, N_SITES)))
    return {"x1": pairs(_S1), "x2": pairs(_S2), "layer": int(au[_LAYER0].sum())}

def _sign(v, tol=1e-9):
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
