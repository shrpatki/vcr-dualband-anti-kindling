from collections import namedtuple

import numpy as np

from .params import N, N_SITES, STIM_DT_S, neuron_positions, site_positions

Channel = namedtuple('Channel', ['name', 'f_best', 'bandwidth', 'rf_width'])

FA1 = Channel('FA1', 30.0, 0.6, 0.05)
FA2 = Channel('FA2', 250.0, 0.6, 0.15)
FOCAL_250 = Channel('focal_250Hz', 250.0, 0.6, 0.05)
BROAD_30 = Channel('broad_30Hz', 30.0, 0.6, 0.15)

ANATOMICAL = (FA1, FA2)

ENVELOPES = ('box', 'gauss', 'dc_gauss', 'sine', 'rsine')


def tuning(f, channel):
    return np.exp(-((np.log2(f) - np.log2(channel.f_best)) ** 2) / (2 * channel.bandwidth ** 2))


def receptive_field(width, n=N, n_sites=N_SITES):
    """Site to neuron weights, normalized so that every site delivers n / n_sites in total."""
    d = neuron_positions(n)[:, None] - site_positions(n_sites)[None, :]
    w = np.exp(-(d ** 2) / (2 * width ** 2))
    return w / w.sum(axis=0, keepdims=True) * (n / n_sites)


def receptive_fields(channels, n=N, n_sites=N_SITES):
    return np.stack([receptive_field(ch.rf_width, n, n_sites) for ch in channels], axis=0)


def envelope_and_carrier(kind, tt, dur, phase):
    if kind not in ENVELOPES:
        raise ValueError(f'unknown envelope {kind}')
    gaussian = np.exp(-((tt - dur / 2) ** 2) / (2 * (dur / 4) ** 2))
    env = gaussian if kind in ('gauss', 'dc_gauss') else np.ones_like(tt)
    if kind == 'dc_gauss':
        carrier = np.ones_like(tt)
    elif kind == 'sine':
        carrier = np.sin(phase)
    else:
        carrier = np.abs(np.sin(phase))
    return env, carrier


def burst_forcing(events, t_grid, channels=ANATOMICAL, n_sites=N_SITES):
    """Input from each channel at each site, shape (n_channels, n_sites, len(t_grid))."""
    n_t = len(t_grid)
    forcing = np.zeros((len(channels), n_sites, n_t))
    for ev in events:
        i0 = max(int(round(ev['onset'] / STIM_DT_S)), 0)
        i1 = min(int(round((ev['onset'] + ev['dur']) / STIM_DT_S)), n_t)
        if i1 <= i0:
            continue
        tt = t_grid[i0:i1] - ev['onset']
        f = ev['freq'](tt)
        if len(tt) > 1:
            phase = 2 * np.pi * np.concatenate(([0.0], np.cumsum((f[1:] + f[:-1]) / 2 * STIM_DT_S)))
        else:
            phase = np.zeros_like(tt)
        env, carrier = envelope_and_carrier(ev['env'], tt, ev['dur'], phase + ev.get('phase0', 0.0))
        amp = ev.get('gain', 1.0) * env * carrier
        for c, ch in enumerate(channels):
            forcing[c, ev['site'], i0:i1] += amp * tuning(f, ch)
    return forcing
