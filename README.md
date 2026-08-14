# ATOMICA — Slice 1: LJ-cluster search benchmark

Compares Random vs Genetic vs Active-learning search for Lennard-Jones cluster
global minima, under an equal budget of local relaxations.

## Setup
    python3 -m pip install -r requirements.txt

## Run
    python3 -m atomica.run --n 13 38 --budget 200 --seeds 5

Outputs `results/convergence_N{n}.png` and per-run JSON.

## Tests
    python3 -m pytest -q

See `docs/superpowers/specs/2026-08-13-atomica-slice1-design.md` for design and roadmap.
