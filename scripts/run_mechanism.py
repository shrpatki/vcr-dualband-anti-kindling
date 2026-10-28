"""Sector coherence and post-burst pauses during the first 20 s of stimulation for every
pattern and control (networks 90-104). The first conditions use the same schedule seeds
as run_coherence.py."""
import pandas as pd
from joblib import Parallel, delayed

from common import parser, sim_range, save
from vcr import params as P
from vcr.experiment import simulate, default_jobs
from vcr.measures import sector_counts, network_coherence, long_intervals
from vcr.stimuli import MAIN, CLINICAL, DOSE_CONTROLS, ENVELOPE_CONTROLS, PHASE

CONTROLS = ['only_30hz', 'two_field_250hz', 'constant_substitution', 'both_focal', 'both_broad']
ALL = MAIN + CLINICAL + DOSE_CONTROLS + ENVELOPE_CONTROLS + CONTROLS + PHASE
WINDOW_S = 20.0
PROTOCOL = (2.0, 60.0, WINDOW_S, 0.0)


def one(k, condition):
    position = ALL.index('dual_band') if condition in PHASE else ALL.index(condition)
    _, arr = simulate(k, condition, position, protocol=PROTOCOL, record_spikes=True,
                      spike_window_ms=WINDOW_S * 1000)
    t, i = arr['spike_t'] / 1000.0, arr['spike_i']
    on = arr['stim_on_ms'] / 1000.0
    summary, _, _ = network_coherence(sector_counts(t, i, on, WINDOW_S, P.N), P.N)
    return dict(condition=condition, sim=k, long_intervals=long_intervals(t, i, on, WINDOW_S, P.N), **summary)


if __name__ == '__main__':
    args = parser(__doc__, '90-104').parse_args()
    rows = Parallel(n_jobs=args.jobs or default_jobs(), backend='loky', verbose=5)(
        delayed(one)(k, c) for k in sim_range(args.sims) for c in ALL)
    save(pd.DataFrame(rows), args.out, 'mechanism.csv')
