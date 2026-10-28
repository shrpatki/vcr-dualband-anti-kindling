"""Response of isolated Hodgkin-Huxley neurons at a sector centre to one burst.

For each phase of the 30 Hz component (0-180 degrees; the rectified component has a
period of 180 degrees) and each of 16 input currents spanning the range used in the
network, the longest inter-spike interval after the burst is recorded. The regular vCR
burst is included for comparison, and both bursts are also applied over a wider range of
input currents (10-20 uA/cm2) to show how the response depends on the distance from the
onset of repetitive firing. Writes results/single_neuron.csv and the voltage traces
used in the figures to results/single_neuron_traces.npz.
"""
import os

import numpy as np
import pandas as pd

from common import RESULTS, save
from vcr import params as P
from vcr.afferents import ANATOMICAL, burst_forcing, receptive_fields
from vcr.neuron import step_neuron
from vcr.stimuli import DUAL_GAIN, T_BURST, constant

ONSET = 0.3        # s
DURATION = 0.7     # s
SITE, NEURON = 2, 40
CURRENTS = np.linspace(P.I0 - P.DELTA_I, P.I0 + P.DELTA_I, 16)
PHASES = np.arange(0, 180, 15)
CURRENT_SWEEP = np.arange(10.0, 20.01, 0.5)


def burst(kind, phase_deg=0.0):
    if kind == 'regular':
        return [dict(site=SITE, onset=ONSET, dur=T_BURST, env='box', freq=constant(250.0))]
    return [dict(site=SITE, onset=ONSET, dur=T_BURST, env='gauss', freq=constant(30.0), gain=DUAL_GAIN,
                 phase0=np.radians(phase_deg)),
            dict(site=SITE, onset=ONSET, dur=T_BURST, env='gauss', freq=constant(250.0), gain=DUAL_GAIN)]


def respond(events, current):
    t = np.arange(0, DURATION, P.STIM_DT_S)
    drive = np.einsum('cs,cst->t', receptive_fields(ANATOMICAL)[:, NEURON, :], burst_forcing(events, t))
    V, m, h, n = -65.0, 0.05, 0.6, 0.32
    n_steps = int(DURATION * 1000 / P.DT)
    trace = np.empty(n_steps)
    spikes = []
    for step in range(n_steps):
        d = drive[min(step // P.STIM_STRIDE, len(drive) - 1)]
        V_new, m, h, n = step_neuron(V, m, h, n, current, 0.0, P.K_STIM * (P.V_STIM - V) * d, P.DT)
        if V_new > 0.0 >= V:
            spikes.append(step * P.DT)
        V = V_new
        trace[step] = V
    return np.array(spikes), trace, drive


def longest_interval_after_onset(spikes):
    s = spikes[spikes > ONSET * 1000 - 50]
    return float(np.diff(s).max()) if len(s) > 1 else np.nan


if __name__ == '__main__':
    rows = []
    for current in CURRENTS:
        spikes, _, _ = respond(burst('regular'), current)
        rows.append(dict(burst='regular', phase=np.nan, current=current,
                         longest_interval_ms=longest_interval_after_onset(spikes)))
        for phase in PHASES:
            spikes, _, _ = respond(burst('dual', phase), current)
            rows.append(dict(burst='dual_band', phase=phase, current=current,
                             longest_interval_ms=longest_interval_after_onset(spikes)))
    for current in CURRENT_SWEEP:
        for name, events in (('regular', burst('regular')), ('dual_band', burst('dual', 0))):
            spikes, _, _ = respond(events, current)
            rows.append(dict(burst=name, phase=0.0 if name == 'dual_band' else np.nan, current=current,
                             longest_interval_ms=longest_interval_after_onset(spikes), sweep='current'))
    save(pd.DataFrame(rows), RESULTS, 'single_neuron.csv')

    traces = {}
    for name, events in (('regular', burst('regular')), ('phase0', burst('dual', 0)), ('phase90', burst('dual', 90))):
        _, trace, drive = respond(events, P.I0)
        traces[f'V_{name}'] = trace[::10].astype(np.float32)
        traces[f'D_{name}'] = drive.astype(np.float32)
    traces['t_ms'] = np.arange(len(traces['V_regular'])) * P.DT * 10
    np.savez_compressed(os.path.join(RESULTS, 'single_neuron_traces.npz'), **traces)
