"""Dual-band burst, regular vCR and jittered vCR in a network of 200 neurons (networks 0-14)."""
from common import parser, sim_range, save
from vcr.experiment import run_conditions

CONDITIONS = ['dual_band', 'regular', 'jittered']

if __name__ == '__main__':
    args = parser(__doc__, '0-14').parse_args()
    df = run_conditions(CONDITIONS, sim_range(args.sims), jobs=args.jobs, n=200)
    save(df, args.out, 'network_size.csv')
