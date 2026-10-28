"""Coherence between network sectors and synchrony-band power (networks 90-104),
and the inter-site coherence of the stimulus itself."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import parser, sim_range, save
from vcr import params as P
from vcr.afferents import burst_forcing
from vcr.experiment import simulate, default_jobs
from vcr.measures import sector_counts, network_coherence, input_coherence
from vcr.stimuli import CONDITIONS, MAIN, CLINICAL

ALL = MAIN + CLINICAL
RECORDED = ['synchronous', 'regular', 'dual_band', 'clinical_replica', 'clinical_dual_band']
WINDOW_S = 20.0


def one(k, condition):
    _, arr = simulate(k, condition, ALL.index(condition), record_spikes=True, spike_window_ms=WINDOW_S * 1000)
    t, i = arr['spike_t'] / 1000.0, arr['spike_i']
    on, off = arr['stim_on_ms'] / 1000.0, arr['stim_off_ms'] / 1000.0
    rows, spectra = [], {}
    for window, t0 in (('baseline', on - WINDOW_S), ('stimulation', on), ('post', off)):
        summary, f, power = network_coherence(sector_counts(t, i, t0, WINDOW_S, P.N), P.N)
        rows.append(dict(condition=condition, sim=k, window=window, **summary))
        spectra[window] = power
    return rows, spectra, f


def stimulus_coherence(seed=3, duration=20.0):
    rows = []
    for name in ALL:
        generator, channels, _ = CONDITIONS[name]
        forcing = burst_forcing(generator(np.random.default_rng(seed), 0.0, duration),
                                np.arange(0, duration, P.STIM_DT_S), channels=channels)
        rows.append(dict(condition=name, input_coherence=input_coherence(forcing)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    args = parser(__doc__, '90-104').parse_args()
    sims = sim_range(args.sims)
    results = Parallel(n_jobs=args.jobs or default_jobs(), backend='loky', verbose=5)(
        delayed(one)(k, c) for k in sims for c in RECORDED)

    rows, keys, power = [], [], []
    for r, spectra, f in results:
        rows.extend(r)
        for window, p in spectra.items():
            keys.append(f"{r[0]['condition']}|{r[0]['sim']}|{window}")
            power.append(p)
    save(pd.DataFrame(rows), args.out, 'coherence.csv')
    np.savez_compressed(f'{args.out}/coherence_spectra.npz', freq=f, keys=np.array(keys),
                        power=np.array(power, dtype=np.float32))
    save(stimulus_coherence(), args.out, 'input_coherence.csv')
