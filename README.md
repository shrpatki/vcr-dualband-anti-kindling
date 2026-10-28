# Dual-band vibrotactile coordinated reset in a spiking network model

Code for the simulations, statistics and figures of the paper "Phase-locked dual-band vibration bursts strengthen anti-kindling by vibrotactile coordinated reset stimulation in a spiking network model".

The model is a ring of 80 Hodgkin-Huxley neurons with Mexican-hat connectivity and spike-timing-dependent plasticity. Vibration bursts at five fingertip sites are turned into a stimulation current by two afferent channels (FA1, best frequency 30 Hz, narrow receptive field; FA2, best frequency 250 Hz, broad receptive field). We compare eleven vibrotactile CR patterns and a set of control conditions on the same simulated networks and measure how much the stimulation weakens synaptic coupling after it ends. Two properties of the dual-band burst are isolated by the controls: the phase of its 30 Hz component against the burst envelope, which determines whether a burst is followed by a pause in firing, and the width of the receptive field through which each component reaches the network.

## Installation

Python 3.11 or newer.

```
pip install -r requirements.txt
pip install -e .
```

The integration loop is compiled with Numba, so the first simulation in each process takes a few extra seconds.

## Repository layout

```
vcr/                 model and analysis code
  params.py          model parameters and protocol durations
  neuron.py          Hodgkin-Huxley kinetics
  network.py         connectivity and plasticity rule
  afferents.py       frequency tuning, receptive fields, stimulus input
  stimuli.py         stimulation schedules and the named conditions
  simulate.py        numerical integration of the network
  experiment.py      running conditions on simulated networks in parallel
  measures.py        outcome measures, coherence and stimulus dose
  stats.py           paired tests, effect sizes, Holm correction, bootstrap
scripts/             one script per study, plus analysis and figures
results/             simulation results used in the paper
figures/             figures as PNG and PDF
```

## Reproducing the paper

The `results/` folder contains the output of every simulation reported in the paper, so the statistics and figures can be regenerated without running any simulations:

```
python scripts/analyze.py
python scripts/make_figures.py
```

`analyze.py` writes the tables to `results/tables/`. `make_figures.py` writes Figures 1 to 9 and S1 to S4 to `figures/`, and accepts figure names as arguments (`python scripts/make_figures.py 6 S2`).

To rerun the simulations, run the study scripts from the repository root, or `sh scripts/run_queue.sh` for all of them in order. Each one writes its result file to `results/`.

| Script | Study | Networks | Approximate run time |
|---|---|---|---|
| `run_main.py` | eleven stimulation patterns | 0-29 | 8 h |
| `run_controls.py` | dose, envelope, component and receptive-field controls | 0-29 and 0-14 | 7 h |
| `run_clinical.py` | clinical-style protocol with five sites | 0-29 | 1.5 h |
| `run_four_sites.py` | clinical-style protocol with four sites | 0-29 | 1.5 h |
| `run_coherence.py` | network coherence and stimulus coherence | 90-104 | 1.5 h |
| `run_mechanism.py` | sector coherence and post-burst pauses for every condition | 90-104 | 1 h |
| `run_confirmation.py` | regular vCR and the dual-band burst on fresh networks | 30-59 | 3 h |
| `run_phase.py` | phase of the 30 Hz component (`--sweep` for 45 and 135 degrees) | 0-59 | 6 h |
| `run_robustness.py` | single parameter changes (`--changes`, `--name`) | 0-29 | 9 h |
| `run_network_size.py` | 200-neuron network | 0-14 | 8 h |
| `run_gain_sweep.py` | component amplitude and second-component frequency | 0-14 | 5.5 h |
| `run_calibration.py` | stimulation strength | 0-4 | 0.5 h |
| `run_single_neuron.py` | response of isolated neurons to one burst | none | 2 min |
| `run_example.py` | spike rasters and weight matrices for one network | 0 | 10 min |

Run times are for five worker processes on a desktop machine. Every script accepts `--sims` (for example `--sims 0-4` for a quick check), `--jobs` and `--out`.

## Reproducibility

Each index k defines one network. Its input currents and initial weights are drawn with seed 1000 + k and its initial state with seed 5000 + 1000 k, so every condition in a study runs on exactly the same networks and all comparisons are paired. The random parts of a stimulation schedule (site order, onset jitter, chaotic maps) use seed 2000 + 1000 k + p, where p is the position of the condition in the list the study runs. Because of this, the order of the condition lists in `vcr/stimuli.py` and in the study scripts should not be changed if you want to reproduce the published numbers.

The rows for smooth vCR in `main.csv` come from a separate run on the same networks, made when that pattern was changed to a rectified waveform. Its schedule contains no random elements, so `run_main.py` reproduces it on the same networks.

The network is chaotic, so results are only reproduced exactly with the same floating point operations. They were produced with the package versions in `requirements.txt` on Windows 11 with an Intel processor. Other platforms or library versions can give slightly different values for individual networks.

## Results files

| File | Contents |
|---|---|
| `main.csv` | one row per pattern and network |
| `controls.csv` | control conditions, with the study in the `set` column |
| `clinical.csv` | clinical protocol comparison |
| `coherence.csv` | network coherence and synchrony-band power in three 20 s windows |
| `coherence_spectra.npz` | population spike-count spectra |
| `input_coherence.csv` | inter-site coherence of the stimulus |
| `robustness.csv` | parameter changes, with the change in the `change` column |
| `network_size.csv` | 200-neuron network |
| `gain_sweep.csv` | component amplitude and second-component frequency |
| `calibration.csv` | synchrony during 30 s of stimulation for each stimulation strength |
| `confirmation.csv` | regular vCR and the dual-band burst on networks 30-59 |
| `phase.csv`, `phase_sweep.csv` | phase variants of the dual-band burst |
| `four_sites.csv` | clinical-style protocol with four stimulation sites |
| `mechanism.csv` | sector coherence and post-burst pauses for every condition |
| `robustness_current.csv`, `robustness_bandwidth.csv` | input current and afferent tuning bandwidth |
| `single_neuron.csv`, `single_neuron_traces.npz` | isolated neurons receiving one burst |
| `traces.npz` | order parameter and coupling in 1 s bins, keyed `condition/R` and `condition/Cav` |
| `example/` | spike times and weight matrices of network 0 |
| `tables/` | the statistics reported in the paper, written by `analyze.py` |

Columns: `R_baseline`, `R_on` and `R_post` are the order parameter averaged over the last 5 s of baseline, stimulation and the post-stimulation period. `Cav_*` is the mean signed coupling over the same windows (lower means stronger anti-kindling), `Cav_weighted_*` the coupling weighted by connection strength and `saturated_*` the fraction of weights at 0 or 1. `held` is true when `R_post` is below 90% of `R_baseline`.
