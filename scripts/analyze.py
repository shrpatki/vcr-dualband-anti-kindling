"""Statistics reported in the paper, written to results/tables/."""
import os

import numpy as np
import pandas as pd

from common import RESULTS, save
from vcr import params as P
from vcr.afferents import burst_forcing, receptive_fields
from vcr.measures import dose, sector_input_coherence, own_sector_share
from vcr.network import mexican_hat, coupling_bounds
from vcr.stats import (compare, compare_against, friedman, split_half, bootstrap_first, hold_sensitivity,
                       summarize, holm, tost)
from vcr.stimuli import CONDITIONS, MAIN, SITES

TABLES = os.path.join(RESULTS, 'tables')

EQUIVALENCE_MARGIN = 0.006   # half of the dual-band advantage over the carrier in the main comparison


def load(name):
    path = os.path.join(RESULTS, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def coupling_range(summary, n=P.N):
    lo, hi = coupling_bounds(mexican_hat(n))
    out = summary[['Cav_post_mean']].copy()
    out['percent_of_range'] = (hi - out['Cav_post_mean']) / (hi - lo) * 100
    return out.reset_index()


def main_comparison(main):
    summary = summarize(main)
    save(summary.reset_index(), TABLES, 'main_summary.csv')
    save(coupling_range(summary), TABLES, 'main_coupling_range.csv')

    others = [c for c in MAIN if c != 'dual_band']
    save(compare_against(main, 'dual_band', others), TABLES, 'main_dual_band_vs_all.csv')
    save(pd.DataFrame([compare(main, 'dual_band', 'regular', m) for m in ('R_on', 'R_post', 'Cav_weighted_post')]),
         TABLES, 'main_dual_band_vs_regular_secondary.csv')
    best, half = split_half(main)
    half.insert(0, 'selected_on_odd', best)
    save(half, TABLES, 'main_split_half.csv')
    save(bootstrap_first(main).rename('fraction_first').reset_index().rename(columns={'index': 'condition'}),
         TABLES, 'main_bootstrap.csv')
    save(hold_sensitivity(main).reset_index(), TABLES, 'main_maintained_thresholds.csv')
    stat, p = friedman(main)
    print(f'Friedman chi2 = {stat:.1f}, p = {p:.3g}')
    save(pd.DataFrame([compare(main, 'dual_band', 'regular', 'Cav_on')]), TABLES, 'main_end_of_stimulation.csv')


def controls(main, ctrl):
    """All control comparisons form one Holm family; equivalence by two one-sided tests."""
    reference = main[main['condition'].isin(['dual_band', 'regular'])]
    tables = []
    for name in ('dose', 'envelope'):
        sub = pd.concat([reference, ctrl[ctrl['set'] == name]])
        others = sorted(ctrl.loc[ctrl['set'] == name, 'condition'].unique())
        tables.append(compare_against(sub, 'dual_band', others, correct=False).assign(set=name))
        if name == 'envelope':
            save(pd.DataFrame([compare(sub, c, 'regular') for c in others]), TABLES, 'controls_envelope_vs_regular.csv')
    for name in ('components', 'fields'):
        sub = ctrl[ctrl['set'] == name]
        others = [c for c in sub['condition'].unique() if c not in ('dual_band', 'clinical_replica')]
        save(summarize(sub).reset_index(), TABLES, f'controls_{name}_summary.csv')
        t = compare_against(sub, 'dual_band', others, correct=False).assign(set=name)
        t['p_equivalence'] = [tost(sub, 'dual_band', c, EQUIVALENCE_MARGIN) for c in others]
        tables.append(t)
    table = pd.concat(tables, ignore_index=True)
    table['holm'] = holm(table['p'].values)
    save(table, TABLES, 'controls.csv')


def clinical(main, clin):
    primary = compare(clin, 'clinical_dual_band', 'clinical_replica')
    secondary = pd.DataFrame([compare(clin, 'clinical_dual_band', 'clinical_replica', m)
                              for m in ('R_post', 'R_on', 'Cav_weighted_post')])
    secondary['holm'] = holm(secondary['p'].values)
    both = pd.concat([clin, main[main['condition'].isin(['regular', 'dual_band'])]])
    context = pd.DataFrame([compare(both, 'clinical_dual_band', 'regular'),
                            compare(both, 'clinical_replica', 'regular'),
                            compare(both, 'clinical_dual_band', 'dual_band')])
    save(summarize(clin).reset_index(), TABLES, 'clinical_summary.csv')
    save(pd.concat([pd.DataFrame([primary]), secondary, context], ignore_index=True), TABLES, 'clinical_tests.csv')


def coherence(coh):
    metrics = ['sector_coherence', 'sector_coherence_sync_band', 'sync_band_power',
               'sync_peak_hz', 'mean_rate_hz']
    summary = coh.groupby(['window', 'condition'])[metrics].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    save(summary.reset_index(), TABLES, 'coherence_summary.csv')

    def test(window, metric, a='dual_band', b='regular'):
        w = coh[coh['window'] == window]
        return dict(window=window, **compare(w, a, b, metric))

    primary = test('stimulation', 'sector_coherence')
    secondary = pd.DataFrame([test('stimulation', 'sync_band_power'),
                              test('stimulation', 'sector_coherence_sync_band'),
                              test('stimulation', 'sector_coherence', 'clinical_dual_band', 'clinical_replica'),
                              test('post', 'sync_band_power')])
    secondary['holm'] = holm(secondary['p'].values)
    descriptive = pd.DataFrame([test('post', 'sector_coherence'), test('post', 'sector_coherence_sync_band')])
    save(pd.concat([pd.DataFrame([primary]), secondary, descriptive], ignore_index=True), TABLES, 'coherence_tests.csv')


def robustness(main, rob):
    extra = [load(name) for name in ('robustness_current.csv', 'robustness_bandwidth.csv')]
    rob = pd.concat([rob] + [e for e in extra if e is not None], ignore_index=True)
    rows = []
    for change, sub in [('reference', main)] + list(rob.groupby('change')):
        for metric in ('Cav_post', 'R_on', 'R_post'):
            rows.append(dict(change=change, **compare(sub, 'dual_band', 'regular', metric)))
    table = pd.DataFrame(rows)
    cav = table[(table['metric'] == 'Cav_post') & (table['change'] != 'reference')]
    table.loc[cav.index, 'holm'] = holm(cav['p'].values)
    save(table, TABLES, 'robustness.csv')


def network_size(size):
    primary = compare(size, 'dual_band', 'regular')
    secondary = pd.DataFrame([compare(size, 'dual_band', 'regular', m)
                              for m in ('Cav_weighted_post', 'R_on', 'R_post')])
    secondary['holm'] = holm(secondary['p'].values)
    jittered = compare(size, 'dual_band', 'jittered')
    save(pd.concat([pd.DataFrame([primary]), secondary, pd.DataFrame([jittered])], ignore_index=True),
         TABLES, 'network_size_tests.csv')
    summary = summarize(size)
    save(summary.reset_index(), TABLES, 'network_size_summary.csv')
    save(coupling_range(summary, n=200), TABLES, 'network_size_coupling_range.csv')


def gain_sweep(sweep, ctrl):
    save(sweep.pivot_table(index='gain', columns='f2', values='Cav_post', aggfunc='mean').reset_index(),
         TABLES, 'gain_sweep_surface.csv')
    carrier = ctrl[(ctrl['set'] == 'components') & (ctrl['condition'] == 'carrier_only')]
    rows = []
    for gain, sub in sweep[sweep['f2'] == 250.0].groupby('gain'):
        both = pd.concat([sub.assign(condition='sweep'), carrier])
        rows.append(dict(gain=gain, **compare(both, 'sweep', 'carrier_only')))
    save(pd.DataFrame(rows), TABLES, 'gain_sweep_vs_carrier.csv')


def calibration(cal):
    save(cal.groupby(['K_stim', 'condition']).agg(R_baseline=('R_baseline', 'mean'), R_on=('R_on', 'mean'),
                                                  R_on_sd=('R_on', 'std')).reset_index(), TABLES, 'calibration.csv')


def phase(main, ph, conf):
    """Phase of the 30 Hz component, on networks 0-29 and on the fresh set 30-59."""
    sweep = load('phase_sweep.csv')
    both = pd.concat([main[main['condition'].isin(['regular', 'dual_band'])], conf, ph]
                     + ([sweep] if sweep is not None else []))
    variants = [c for c in both['condition'].unique() if c.startswith('dual_band_')]
    rows = []
    for label, sims in (('0-29', range(0, 30)), ('30-59', range(30, 60)), ('0-59', range(0, 60))):
        sub = both[both['sim'].isin(sims)]
        pairs = [('dual_band', 'regular')] + [(v, b) for v in variants for b in ('regular', 'dual_band')]
        for a, b in pairs:
            if not (set(sub[sub['condition'] == a]['sim']) & set(sub[sub['condition'] == b]['sim'])):
                continue
            for metric in ('Cav_post', 'Cav_on', 'R_on', 'Cav_weighted_post'):
                rows.append(dict(sims=label, **compare(sub, a, b, metric)))
    save(pd.DataFrame(rows), TABLES, 'phase_tests.csv')
    save(summarize(both[both['sim'] >= 30]).reset_index(), TABLES, 'confirmation_summary.csv')
    save(summarize(both).reset_index(), TABLES, 'phase_summary.csv')


def four_sites(four):
    rows = [compare(four, 'clinical_dual_band_4', 'clinical_replica_4', m)
            for m in ('Cav_post', 'Cav_weighted_post', 'R_post', 'R_on')]
    save(pd.DataFrame(rows), TABLES, 'four_sites_tests.csv')
    save(summarize(four).reset_index(), TABLES, 'four_sites_summary.csv')


def mechanism(mech, seeds=(9000, 9001, 9002), duration=20.0):
    """Per-condition sector input coherence, own-sector share and network measures."""
    t = np.arange(0, duration, P.STIM_DT_S)
    rows = []
    for name in mech['condition'].unique():
        generator, channels, _ = CONDITIONS[name]
        rf = receptive_fields(channels)
        forcings = [burst_forcing(generator(np.random.default_rng(s), 0.0, duration), t, channels=channels)
                    for s in seeds]
        rows.append(dict(condition=name,
                         sector_input_coherence=np.mean([sector_input_coherence(f, rf) for f in forcings]),
                         own_sector_share=own_sector_share(forcings[0], rf)))
    stim = pd.DataFrame(rows).set_index('condition')
    net = mech.groupby('condition')[['sector_coherence', 'long_intervals', 'sync_band_power']].agg(['mean', 'std'])
    net.columns = ['_'.join(c) for c in net.columns]
    save(stim.join(net).reset_index(), TABLES, 'mechanism.csv')


def stimulus_dose(seeds=(9000, 9001, 9002), duration=20.0):
    t = np.arange(0, duration, P.STIM_DT_S)
    rows = []
    for name, (generator, channels, _) in CONDITIONS.items():
        n_sites = SITES.get(name, P.N_SITES)
        values = [dose(burst_forcing(generator(np.random.default_rng(s), 0.0, duration), t, channels=channels,
                                     n_sites=n_sites), receptive_fields(channels, n_sites=n_sites)) for s in seeds]
        power, drive = np.mean(values, axis=0)
        rows.append(dict(condition=name, power=power, mean_drive=drive))
    table = pd.DataFrame(rows).set_index('condition')
    table['relative_power'] = table['power'] / table.loc['regular', 'power']
    table['relative_mean_drive'] = table['mean_drive'] / table.loc['regular', 'mean_drive']
    save(table.reset_index(), TABLES, 'dose.csv')


ANALYSES = [
    (main_comparison, ['main.csv']),
    (controls, ['main.csv', 'controls.csv']),
    (clinical, ['main.csv', 'clinical.csv']),
    (coherence, ['coherence.csv']),
    (robustness, ['main.csv', 'robustness.csv']),
    (network_size, ['network_size.csv']),
    (gain_sweep, ['gain_sweep.csv', 'controls.csv']),
    (calibration, ['calibration.csv']),
    (phase, ['main.csv', 'phase.csv', 'confirmation.csv']),
    (four_sites, ['four_sites.csv']),
    (mechanism, ['mechanism.csv']),
]

if __name__ == '__main__':
    for analysis, inputs in ANALYSES:
        frames = [load(name) for name in inputs]
        if any(frame is None for frame in frames):
            print('skipped', analysis.__name__, 'without', [n for n, f in zip(inputs, frames) if f is None])
            continue
        analysis(*frames)
    stimulus_dose()
