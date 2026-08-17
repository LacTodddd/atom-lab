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
