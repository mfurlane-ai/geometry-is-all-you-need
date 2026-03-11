# Geometry Is All You Need

Simulation code for the paper:
**"Geometry Is All You Need: Physical 3D Topology as a Foundation
for Neuromorphic AI Architecture"**

Matthew Furlane & Claude (Anthropic) — Draft v0.8, March 2026

## What this is
A NumPy simulation of a 9×9×9 cubic lattice neural network testing
two core predictions: semantic clustering in physical space, and
geometric activation sparsity.

## Results
- Clustering ratio: 7.52 (spatial reg) vs null hypothesis 1.0
- Activation sparsity: 0.81% vs 38.7% flat baseline
- Connections used: 0.37% of equivalent flat network
- Accuracy: 100% — identical to flat baseline

## Run it
pip install numpy matplotlib
python lattice_sim_9x9x9.py

## Files
- lattice_sim_9x9x9.py — main publication simulation
- lattice_sim_5x5x5.py — sanity check simulation
- simulation_results_9x9x9.json — raw numerical results
- fig9_*.png — all generated figures
