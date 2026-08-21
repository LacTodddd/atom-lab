"""P5 literature world: generate a synthetic 'paper' (a summary of a study over an explored
subset of the Cu-Au 924 orderings) plus a ground-truth label. Reuses P4's critic_world unchanged."""
import numpy as np
from atomica.critic_world import build_world, features, FEATURE_NAMES

def make_paper(world, seed, min_frac=0.3, max_frac=0.7):
    rng = np.random.default_rng(seed)
    F, E, configs = world["features"], world["energy"], world["configs"]
    axis = str(rng.choice(FEATURE_NAMES))
    zc = F[axis].astype(float)
    frac = float(rng.uniform(min_frac, max_frac))
    gap_high = bool(rng.integers(2))          # True -> explored is the LOW side, gap is HIGH
    if gap_high:
        cut = np.quantile(zc, frac)
        explored = zc <= cut
    else:
        cut = np.quantile(zc, 1.0 - frac)
        explored = zc >= cut
    gap = ~explored
    ei = np.where(explored)[0]
    best_local = int(ei[np.argmin(E[ei])])
    best_energy = float(E[best_local])
    # boundary_trend: min energy per DISTINCT axis value present in the explored subset,
    # ordered far-from-gap -> near-the-gap. (Quantile sub-bands collapse on discrete features.)
    vals = sorted(int(v) for v in set(zc[explored]))
    order = vals if gap_high else vals[::-1]
    trend = []
    for v in order:
        m = explored & (zc == v)
        trend.append(float(E[m].min()))
    gap_indices = np.where(gap)[0]
    gap_best = float(E[gap_indices].min()) if len(gap_indices) else float(best_energy)
    better_in_gap = bool(len(gap_indices) > 0 and gap_best < best_energy - 1e-9)
    studied_side = "low" if gap_high else "high"
    return {"axis": axis,
            "region": f"orderings with {axis} in the {studied_side} range",
            "n_explored": int(explored.sum()),
            "best_config": list(configs[best_local]),
            "best_energy": best_energy,
            "boundary_trend": trend,
            "gap_side": "high" if gap_high else "low",
            "better_in_gap": better_in_gap,
            "seed": int(seed)}

def generate_papers(world, n_papers, seed0=0, min_frac=0.3, max_frac=0.7, min_count=40):
    papers, seed = [], seed0
    n = len(world["energy"])
    while len(papers) < n_papers:
        p = make_paper(world, seed, min_frac, max_frac)
        seed += 1
        if p["n_explored"] < min_count or (n - p["n_explored"]) < min_count or len(p["boundary_trend"]) < 2:
            continue                        # explored or gap too small, or trend too short to extrapolate
        papers.append(p)
    return papers
