"""Dual-band burst with the component amplitude and second-component frequency varied (networks 0-14)."""
from functools import partial

import pandas as pd
from joblib import Parallel, delayed

from common import parser, sim_range, save
from vcr.experiment import simulate, default_jobs
from vcr.stimuli import dual_band

GAINS = (0.2, 0.4, 0.6, 0.8, 1.0)
SECOND_FREQUENCIES = (60.0, 120.0, 250.0)


def one(k, position, gain, f2):
    row = simulate(k, 'dual_band', position, schedule=partial(dual_band, gain=gain, f2=f2))
    row.update(gain=gain, f2=f2)
    return row


if __name__ == '__main__':
    args = parser(__doc__, '0-14').parse_args()
    grid = [(g, f) for g in GAINS for f in SECOND_FREQUENCIES]
    rows = Parallel(n_jobs=args.jobs or default_jobs(), backend='loky', verbose=5)(
        delayed(one)(k, p, g, f) for k in sim_range(args.sims) for p, (g, f) in enumerate(grid))
    save(pd.DataFrame(rows), args.out, 'gain_sweep.csv')
