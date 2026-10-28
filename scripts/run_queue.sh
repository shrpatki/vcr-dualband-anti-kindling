#!/bin/sh
# Runs every study of the paper in order. Each script also runs on its own.
set -e
export PYTHONPATH="$(pwd)"
J=${JOBS:-18}

python scripts/run_main.py --jobs $J
python scripts/run_controls.py --jobs $J
python scripts/run_clinical.py --jobs $J
python scripts/run_four_sites.py --jobs $J
python scripts/run_coherence.py --jobs $J
python scripts/run_mechanism.py --jobs $J
python scripts/run_confirmation.py --jobs $J
python scripts/run_phase.py --jobs $J
python scripts/run_phase.py --sweep --sims 0-29 --jobs $J
python scripts/run_robustness.py --jobs $J --changes "K_stim 0.20,K_stim 0.50,plasticity rate x0.5,plasticity rate x2,a2 1.5,a2 2.5"
python scripts/run_robustness.py --jobs $J --changes "input current 14" --name robustness_current.csv
python scripts/run_robustness.py --jobs $J --sims 0-14 --changes "bandwidth 1.0,bandwidth 1.5" --name robustness_bandwidth.csv
python scripts/run_network_size.py --jobs $J
python scripts/run_gain_sweep.py --jobs $J
python scripts/run_calibration.py --jobs $J
python scripts/run_single_neuron.py
python scripts/run_example.py --jobs 5

python scripts/analyze.py
python scripts/make_figures.py
