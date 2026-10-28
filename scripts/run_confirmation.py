"""Regular vCR and the dual-band burst on networks that were not used for any parameter
choice (networks 30-59), with R and Cav traces."""
import pandas as pd
from joblib import Parallel, delayed

from common import parser, sim_range, save, save_traces
from vcr.experiment import simulate, default_jobs
from vcr.stimuli import MAIN

CONDITIONS = ['regular', 'dual_band']


def one(k, condition):
    row, trace = simulate(k, condition, MAIN.index(condition), traces=True)
    return row, (condition, k, trace)


if __name__ == '__main__':
    args = parser(__doc__, '30-59').parse_args()
    results = Parallel(n_jobs=args.jobs or default_jobs(), backend='loky', verbose=5)(
        delayed(one)(k, c) for k in sim_range(args.sims) for c in CONDITIONS)
    save(pd.DataFrame([row for row, _ in results]), args.out, 'confirmation.csv')
    save_traces(args.out, [trace for _, trace in results])
