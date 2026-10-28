"""Clinical-style patterns with four stimulation sites and onset jitter of +-19.6 ms, as in
the clinical protocol (networks 0-29)."""
from common import parser, sim_range, save
from vcr.experiment import run_conditions
from vcr.stimuli import FOUR_SITE_CLINICAL

if __name__ == '__main__':
    args = parser(__doc__, '0-29').parse_args()
    df = run_conditions(FOUR_SITE_CLINICAL, sim_range(args.sims), jobs=args.jobs)
    save(df, args.out, 'four_sites.csv')
