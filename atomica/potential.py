import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import FIRE

# Untruncated LJ so energies match reference minima (rc large, no smoothing).
_CALC_KW = dict(epsilon=1.0, sigma=1.0, rc=500.0, smooth=False)

def make_atoms(positions: np.ndarray) -> Atoms:
    positions = np.asarray(positions, dtype=float)
    atoms = Atoms(f"H{len(positions)}", positions=positions)
    atoms.calc = LennardJones(**_CALC_KW)
    return atoms

def relax(positions: np.ndarray, fmax: float = 1e-3, steps: int = 300):
    atoms = make_atoms(positions)
    opt = FIRE(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return atoms.get_positions(), float(atoms.get_potential_energy())
