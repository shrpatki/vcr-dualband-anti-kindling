"""Dual-band burst with the 30 Hz component shifted against the burst envelope.

Default: shifted by 90 degrees and drawn at random in every burst, on networks 0-59.
With --sweep: shifted by 45 and 135 degrees. Every variant uses the schedule of
dual_band in run_main.py, so it differs from the phase-locked burst only in that phase.
"""
import pandas as pd
from joblib import Parallel, delayed

from common import parser, sim_range, save, save_traces
from vcr.experiment import simulate, default_jobs
from vcr.stimuli import MAIN, PHASE, PHASE_SWEEP


def one(k, condition):
    row, trace = simulate(k, condition, MAIN.index('dual_band'), traces=True)
    return row, (condition, k, trace)


if __name__ == '__main__':
    p = parser(__doc__, '0-59')
    p.add_argument('--sweep', action='store_true', help='shifts of 45 and 135 degrees instead')
    args = p.parse_args()
    conditions = PHASE_SWEEP if args.sweep else PHASE
    results = Parallel(n_jobs=args.jobs or default_jobs(), backend='loky', verbose=5)(
        delayed(one)(k, c) for k in sim_range(args.sims) for c in conditions)
    save(pd.DataFrame([row for row, _ in results]), args.out,
         'phase_sweep.csv' if args.sweep else 'phase.csv')
    save_traces(args.out, [trace for _, trace in results])
