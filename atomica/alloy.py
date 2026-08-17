import numpy as np
from ase.build import bulk

N_SITES = 12
N_AU = 6
A_LATTICE = 3.85  # Vegard mean of Cu (3.615) and Au (4.078)

def build_lattice(n_sites=N_SITES):
    at = bulk("Cu", "fcc", a=A_LATTICE, cubic=True).repeat((1, 1, 3))
    assert len(at) == n_sites
    return at

def config_symbols(config, n_sites=N_SITES):
    symbols = ["Cu"] * n_sites
    for i in config:
        symbols[i] = "Au"
    return symbols

_CALC = None
def _calc():
    global _CALC
    if _CALC is None:
        from mace.calculators import mace_mp
        _CALC = mace_mp(model="small", dispersion=False,
                        default_dtype="float64", device="cpu")
    return _CALC

def evaluate(config, n_sites=N_SITES):
    at = build_lattice(n_sites)
    at.symbols = config_symbols(config, n_sites)
    at.calc = _calc()
    return float(at.get_potential_energy())

_NBR = None  # directed (i, j) first-nearest-neighbour index arrays for the fixed lattice
def _neighbours(cutoff=2.9):
    # FCC first-NN distance is a/sqrt(2) ~= 2.72 A; 2.9 A captures first NN only (second is 3.85).
    global _NBR
    if _NBR is None:
        from ase.neighborlist import neighbor_list
        i, j = neighbor_list("ij", build_lattice(), cutoff)
        _NBR = (np.asarray(i), np.asarray(j))
    return _NBR

def sro_descriptor(config, n_sites=N_SITES):
    is_au = np.zeros(n_sites, dtype=bool)
    is_au[list(config)] = True
    i, j = _neighbours()
    ai, aj = is_au[i], is_au[j]
    # directed bonds: each undirected bond counted twice -> divide by 2
    counts = np.array([np.sum(ai & aj), np.sum(ai ^ aj), np.sum(~ai & ~aj)], dtype=float) / 2.0
    total = counts.sum()
    return counts / total if total > 0 else counts
