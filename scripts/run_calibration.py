"""Synchrony during 30 s of stimulation as a function of K_stim (networks 0-4)."""
import pandas as pd
from joblib import Parallel, delayed

from common import parser, sim_range, save
from vcr.experiment import simulate, default_jobs

K_VALUES = (0.10, 0.20, 0.35, 0.50, 0.70, 1.00)
CONDITIONS = ('synchronous', 'regular', 'dual_band')
PROTOCOL = (2.0, 60.0, 30.0, 1.0)


def one(k, position, condition, k_stim):
    row = simulate(k, condition, position, protocol=PROTOCOL, k_stim=k_stim)
    row['K_stim'] = k_stim
    return row


if __name__ == '__main__':
    args = parser(__doc__, '0-4').parse_args()
    grid = [(kv, c) for kv in K_VALUES for c in CONDITIONS]
    rows = Parallel(n_jobs=args.jobs or default_jobs(), backend='loky', verbose=5)(
        delayed(one)(k, p, c, kv) for k in sim_range(args.sims) for p, (kv, c) in enumerate(grid))
    save(pd.DataFrame(rows), args.out, 'calibration.csv')
