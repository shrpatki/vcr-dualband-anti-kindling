"""Spike rasters and weight matrices of one network for the supplementary figures."""
import os

import numpy as np
from joblib import Parallel, delayed

from common import parser, sim_range
from vcr.experiment import simulate
from vcr.stimuli import MAIN

CONDITIONS = ['synchronous', 'regular', 'dual_band', 'dual_band_shifted', 'smooth']
DTYPES = {'spike_t': np.float32, 'spike_i': np.int16, 'c_baseline': np.float32,
          'c_stim': np.float32, 'c_post': np.float32, 'stim_on_ms': np.float64,
          'stim_off_ms': np.float64}


def one(k, condition, out):
    position = MAIN.index('dual_band' if condition.startswith('dual_band_') else condition)
    _, arr = simulate(k, condition, position, record_spikes=True, spike_capacity=400000)
    stored = {key: np.asarray(arr[key]).astype(dtype) for key, dtype in DTYPES.items()}
    np.savez_compressed(os.path.join(out, f'example_{condition}.npz'), **stored)
    return condition


if __name__ == '__main__':
    args = parser(__doc__, '0').parse_args()
    out = os.path.join(args.out, 'example')
    os.makedirs(out, exist_ok=True)
    k = sim_range(args.sims)[0]
    print(Parallel(n_jobs=args.jobs or len(CONDITIONS))(delayed(one)(k, c, out) for c in CONDITIONS))
