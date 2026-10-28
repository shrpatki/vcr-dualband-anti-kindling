"""Control conditions for the dual-band burst.

dose        carrier at the dual-band power and mean drive (networks 0-29)
envelope    single 250 Hz Gaussian component at matched power and mean drive (0-29)
components  single components, two receptive fields, constant substitution, clinical-style replica (0-14)
fields      both components through the focal or the broad receptive field (0-14)
"""
import pandas as pd

from common import parser, sim_range, save
from vcr.experiment import run_conditions
from vcr.stimuli import DOSE_CONTROLS, ENVELOPE_CONTROLS, COMPONENT_CONTROLS, FIELD_CONTROLS

SETS = {
    'dose': (DOSE_CONTROLS, '0-29'),
    'envelope': (ENVELOPE_CONTROLS, '0-29'),
    'components': (COMPONENT_CONTROLS, '0-14'),
    'fields': (FIELD_CONTROLS, '0-14'),
}

if __name__ == '__main__':
    p = parser(__doc__, None)
    p.add_argument('--set', choices=list(SETS) + ['all'], default='all')
    args = p.parse_args()
    names = list(SETS) if args.set == 'all' else [args.set]
    frames = []
    for name in names:
        conditions, default_sims = SETS[name]
        df = run_conditions(conditions, sim_range(args.sims or default_sims), jobs=args.jobs)
        df.insert(0, 'set', name)
        frames.append(df)
    name = 'controls.csv' if args.set == 'all' else f'controls_{args.set}.csv'
    save(pd.concat(frames, ignore_index=True), args.out, name)
