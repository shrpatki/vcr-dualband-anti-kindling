"""Dual-band burst and regular vCR under single parameter changes (networks 0-29).
The reference condition is taken from main.csv."""
import pandas as pd
from joblib import Parallel, delayed

from common import parser, sim_range, save
from vcr import params as P
from vcr.afferents import FA1, FA2
from vcr.experiment import simulate, default_jobs
from vcr.stimuli import MAIN

CHANGES = {
    'K_stim 0.20': dict(k_stim=0.20),
    'K_stim 0.50': dict(k_stim=0.50),
    'plasticity rate x0.5': dict(eta=P.ETA * 0.5),
    'plasticity rate x2': dict(eta=P.ETA * 2.0),
    'a2 1.5': dict(a2=1.5),
    'a2 2.5': dict(a2=2.5),
    'bandwidth 1.0': dict(channels=(FA1._replace(bandwidth=1.0), FA2._replace(bandwidth=1.0))),
    'bandwidth 1.5': dict(channels=(FA1._replace(bandwidth=1.5), FA2._replace(bandwidth=1.5))),
    'input current 14': dict(i0=14.0),
}


def one(k, condition, change):
    row = simulate(k, condition, MAIN.index(condition), **CHANGES[change])
    row['change'] = change
    return row


if __name__ == '__main__':
    p = parser(__doc__, '0-29')
    p.add_argument('--changes', default=','.join(CHANGES), help='comma-separated keys of CHANGES')
    p.add_argument('--name', default='robustness.csv')
    args = p.parse_args()
    changes = args.changes.split(',')
    rows = Parallel(n_jobs=args.jobs or default_jobs(), backend='loky', verbose=5)(
        delayed(one)(k, c, ch) for ch in changes for k in sim_range(args.sims) for c in ('dual_band', 'regular'))
    save(pd.DataFrame(rows), args.out, args.name)
