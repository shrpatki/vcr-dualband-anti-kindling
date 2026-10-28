"""Running one condition on one simulated network, and running many in parallel.

Network k is the same for every condition: its input currents and initial weights
are drawn with seed 1000 + k and its initial state with seed 5000 + 1000 k. The
stimulation schedule is drawn with seed 2000 + 1000 k + p, where p is the position
of the condition in the list that a study runs.
"""
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import params as P
from .afferents import burst_forcing, receptive_fields
from .measures import outcome_measures
from .network import mexican_hat, synaptic_reversal
from .simulate import run_protocol, weighted_signed_weight
from .stimuli import CONDITIONS, SITES


def make_network(k, n=P.N, i0=P.I0):
    rng = np.random.default_rng(1000 + k)
    I_drive = rng.uniform(i0 - P.DELTA_I, i0 + P.DELTA_I, n)
    c0 = rng.normal(0.5, 0.01, (n, n))
    np.fill_diagonal(c0, 0.0)
    c0 = np.clip(c0, 0.0, P.C_MAX)
    return I_drive, c0


def saturation(c, M, eps=1e-6):
    connected = M != 0
    at_bound = connected & ((np.abs(c) < eps) | (np.abs(c - P.C_MAX) < eps))
    return at_bound.sum() / connected.sum()


def simulate(k, condition, position, protocol=P.PROTOCOL, n=P.N, k_stim=P.K_STIM, eta=P.ETA,
             a2=P.A2, i0=P.I0, schedule=None, channels=None, record_spikes=False, spike_window_ms=2000.0,
             spike_capacity=2_000_000, traces=False):
    """Simulate condition `condition` on network k.

    `schedule` and `channels` override the condition's schedule function and
    afferent channels (used by the gain sweep and the bandwidth changes).
    Returns a dict of outcome measures and, if record_spikes or traces is set,
    a dict of arrays. Traces are R and Cav averaged in 1 s bins.
    """
    generator, default_channels, closed_loop = CONDITIONS[condition]
    generator = schedule or generator
    channels = channels or default_channels
    n_sites = SITES.get(condition, P.N_SITES)
    t_equil, t_baseline, t_stim, t_post = protocol

    M = mexican_hat(n, a2=a2)
    V_syn = synaptic_reversal(M)
    I_drive, c0 = make_network(k, n, i0)

    rng = np.random.default_rng(2000 + k * 1000 + position)
    events = generator(rng, 0.0, t_stim)
    forcing = burst_forcing(events, np.arange(0, t_stim, P.STIM_DT_S), channels=channels, n_sites=n_sites)
    rf = receptive_fields(channels, n, n_sites)

    out = run_protocol(I_drive, M, V_syn, c0, forcing, rf,
                       t_equil * 1000.0, t_baseline * 1000.0, t_stim * 1000.0, t_post * 1000.0, P.DT,
                       k_stim, P.V_STIM, P.BETA1, P.BETA2, P.GAMMA1, P.GAMMA2, P.TAU_STDP, eta, P.C_MAX,
                       P.SAVE_STRIDE, 5000 + k * 1000, P.STIM_STRIDE,
                       closed_loop, P.EMA_TAU_MS, P.EMA_THRESHOLD,
                       record_spikes, spike_capacity if record_spikes else 1, spike_window_ms)
    R, cav, on_start, on_end, _, c_base, c_stim, c_post, spike_t, spike_i = out

    t = np.arange(len(R)) * P.DT * P.SAVE_STRIDE / 1000.0
    t_base_end = t_equil + t_baseline
    row = outcome_measures(t, R, cav, t_base_end, t_base_end + t_stim, t_base_end + t_stim + t_post)
    row.update(condition=condition, sim=k,
               Cav_weighted_baseline=weighted_signed_weight(c_base, M),
               Cav_weighted_stim=weighted_signed_weight(c_stim, M),
               Cav_weighted_post=weighted_signed_weight(c_post, M),
               saturated_baseline=saturation(c_base, M),
               saturated_stim=saturation(c_stim, M),
               saturated_post=saturation(c_post, M))
    if traces:
        per_s = int(round(1000.0 / (P.DT * P.SAVE_STRIDE)))
        n_s = len(R) // per_s
        return row, dict(t=np.arange(n_s) + 0.5 - t_base_end,
                         R=R[:n_s * per_s].reshape(n_s, per_s).mean(1),
                         Cav=cav[:n_s * per_s].reshape(n_s, per_s).mean(1))
    if not record_spikes:
        return row
    arrays = dict(R=R, Cav=cav, c_baseline=c_base, c_stim=c_stim, c_post=c_post,
                  spike_t=spike_t, spike_i=spike_i,
                  stim_on_ms=on_start * P.DT, stim_off_ms=on_end * P.DT)
    return row, arrays


def default_jobs():
    return max(1, (os.cpu_count() or 4) - 2)


def run_conditions(conditions, sims, jobs=None, positions=None, **kwargs):
    """Run every condition in `conditions` on every network in `sims`, in parallel.

    The schedule seed uses the position of each condition in `conditions`, or in
    `positions` if given, so a condition can be rerun with the schedule of another list."""
    positions = positions or conditions
    tasks = [delayed(simulate)(k, name, positions.index(name), **kwargs) for k in sims for name in conditions]
    rows = Parallel(n_jobs=jobs or default_jobs(), backend='loky', verbose=5)(tasks)
    return pd.DataFrame(rows)
