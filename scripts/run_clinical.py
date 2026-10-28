"""Clinical-style replica and clinical-style dual-band protocol, 30 paired networks."""
from common import parser, sim_range, save
from vcr.experiment import run_conditions
from vcr.stimuli import CLINICAL

if __name__ == '__main__':
    args = parser(__doc__, '0-29').parse_args()
    df = run_conditions(CLINICAL, sim_range(args.sims), jobs=args.jobs)
    save(df, args.out, 'clinical.csv')
