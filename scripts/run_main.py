"""Eleven stimulation patterns, 30 paired networks each."""
from common import parser, sim_range, save
from vcr.experiment import run_conditions
from vcr.stimuli import MAIN

if __name__ == '__main__':
    args = parser(__doc__, '0-29').parse_args()
    df = run_conditions(MAIN, sim_range(args.sims), jobs=args.jobs)
    save(df, args.out, 'main.csv')
