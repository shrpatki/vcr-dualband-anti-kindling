"""Figures 1-8 and S1-S4 from the files in results/."""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats as st

from common import RESULTS, FIGURES, load_traces
from vcr import params as P
from vcr.afferents import FA1, FA2, ANATOMICAL, burst_forcing, receptive_field, receptive_fields, tuning
from vcr.network import mexican_hat, stdp_kernel, coupling_bounds
from vcr.stimuli import T_CR, T_BURST, synchronous, regular, dual_band, clinical_replica, constant
from vcr.stats import by_sim

TABLES = os.path.join(RESULTS, 'tables')

LABEL = {
    'synchronous': 'Synchronous multi-site control', 'regular': 'Regular vCR', 'jittered': 'Jittered vCR',
    'smooth': 'Smooth vCR', 'interleaved': 'Interleaved-carrier CR', 'chirped': 'Chirped-carrier CR',
    'dual_band': 'Dual-band CR', 'on_off': 'On-off vCR', 'chaotic_order': 'Chaotic-order CR',
    'closed_loop': 'Closed-loop CR', 'chaotic_timing': 'Chaotic-timing CR',
    'clinical_replica': 'Clinical-style replica', 'clinical_dual_band': 'Clinical-style dual-band',
    'dual_band_shifted': 'Dual-band, 30 Hz shifted by 90°',
}
SHORT = {
    'synchronous': 'Synchronous\ncontrol', 'regular': 'Regular\nvCR', 'jittered': 'Jittered\nvCR',
    'smooth': 'Smooth\nvCR', 'interleaved': 'Interleaved\ncarrier', 'chirped': 'Chirped\ncarrier',
    'dual_band': 'Dual-band', 'on_off': 'On-off\nvCR', 'chaotic_order': 'Chaotic\norder',
    'closed_loop': 'Closed-\nloop', 'chaotic_timing': 'Chaotic\ntiming',
    'clinical_replica': 'Clinical\nreplica', 'clinical_dual_band': 'Clinical\ndual-band',
}
COLOR = {
    'synchronous': '#000000', 'regular': '#0072B2', 'jittered': '#009E73', 'smooth': '#DDAA33',
    'interleaved': '#CC79A7', 'chirped': '#999933', 'dual_band': '#D55E00', 'on_off': '#882255',
    'chaotic_order': '#117733', 'closed_loop': '#44AA99', 'chaotic_timing': '#332288',
    'clinical_replica': '#56B4E9', 'clinical_dual_band': '#D55E00',
}
FA1_COLOR, FA2_COLOR = '#6A3D9A', '#D55E00'
MARKER = {
    'synchronous': 'X', 'regular': 'o', 'jittered': 's', 'smooth': 'v', 'interleaved': 'D', 'chirped': '^',
    'dual_band': 'P', 'on_off': 'h', 'chaotic_order': '<', 'closed_loop': '>', 'chaotic_timing': 'd',
    'clinical_replica': 'o', 'clinical_dual_band': 'P',
}
SHIFT_COLOR = '#7F7F7F'

MECH_GROUPS = {
    'single': (['regular', 'jittered', 'chirped', 'chaotic_order', 'on_off', 'closed_loop', 'carrier_at_dual_power',
                'carrier_at_dual_drive', 'gauss250_at_dual_power', 'gauss250_at_dual_drive', 'clinical_replica'],
               COLOR['regular'], 'o', 'One broad 250 Hz component'),
    'locked': (['dual_band', 'constant_substitution', 'both_focal', 'both_broad', 'clinical_dual_band'],
               COLOR['dual_band'], 'P', 'Dual-band, phase-locked 30 Hz'),
    'shifted': (['dual_band_shifted', 'dual_band_random_phase', 'dual_band_phase45', 'dual_band_phase135'],
                SHIFT_COLOR, 'X', 'Dual-band, phase-shifted 30 Hz'),
    'focal250': (['two_field_250hz'], '#009E73', 'D', 'Focal and broad 250 Hz'),
    'other': (['interleaved', 'only_30hz', 'smooth', 'chaotic_timing'], '#CC79A7', 's', 'Other patterns'),
    'sync': (['synchronous'], '#000000', 'v', 'Synchronous control'),
}
MECH_COLOR = {k: g[1] for g in MECH_GROUPS.values() for k in g[0]}
MECH_MARKER = {k: g[2] for g in MECH_GROUPS.values() for k in g[0]}
MECH_LABEL = {g[0][0]: g[3] for g in MECH_GROUPS.values()}
MECH_LEGEND = [g[0][0] for g in MECH_GROUPS.values()]


def mechanism_outcome(conditions):
    """Mean C_av,post of each condition in the study that defines it (synchronous control excluded)."""
    values = {}
    main = data('main.csv')
    for k, v in main.groupby('condition')['Cav_post'].mean().items():
        values[k] = v
    ctrl = data('controls.csv')
    for (_, k), v in ctrl.groupby(['set', 'condition'])['Cav_post'].mean().items():
        if k not in values:
            values[k] = v
    for name in ('clinical.csv', 'phase.csv', 'phase_sweep.csv'):
        path = os.path.join(RESULTS, name)
        if os.path.exists(path):
            d = pd.read_csv(path)
            d = d[d['sim'] < 30]
            for k, v in d.groupby('condition')['Cav_post'].mean().items():
                values[k] = v
    values.pop('synchronous', None)
    return pd.Series(values).reindex([c for c in conditions if c in values])

SHOWN = ['synchronous', 'regular', 'jittered', 'interleaved', 'chirped', 'dual_band']
STAGGERED = ['regular', 'jittered', 'interleaved', 'chirped', 'dual_band']
OTHER = ['smooth', 'on_off', 'chaotic_order', 'closed_loop', 'chaotic_timing']

plt.rcParams.update({
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 300, 'pdf.fonttype': 42,
})


def table(name):
    return pd.read_csv(os.path.join(TABLES, name))


def data(name):
    return pd.read_csv(os.path.join(RESULTS, name))


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGURES, f'{name}.{ext}'), bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def letter(ax, text, x=-0.14, y=1.04):
    ax.text(x, y, text.upper(), transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')


def format_p(p):
    e = int(np.floor(np.log10(p)))
    if p < 1e-10:
        return f'p < $10^{{{e + 1}}}$'
    if p < 0.01:
        m = p / 10 ** e
        if round(m, 1) >= 10:
            m, e = m / 10, e + 1
        return f'p = {m:.1f}$\\times10^{{{e}}}$'
    return f'p = {p:.2g}' if p < 0.1 else f'p = {p:.2f}'


def ci95(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def points_with_mean(ax, key, x, values, rng, jitter=0.12, size=8, alpha=0.35, mean_size=6, lw=1.2):
    hollow = key == 'clinical_dual_band'
    ax.scatter(x + rng.uniform(-jitter, jitter, len(values)), values, s=size, marker=MARKER[key],
               facecolor='none' if hollow else COLOR[key], edgecolor=COLOR[key], linewidth=0.4, alpha=alpha, zorder=2)
    m, lo, hi = ci95(values)
    ax.errorbar(x + 0.32, m, yerr=[[m - lo], [hi - m]], fmt=MARKER[key], color=COLOR[key],
                mfc='white' if hollow else COLOR[key], mec='black' if not hollow else COLOR[key], mew=0.6,
                ms=mean_size, capsize=3, elinewidth=lw, capthick=lw, zorder=4)


def paired_lines(ax, a, b, key_a, key_b, label_a, label_b):
    for s in a.index:
        lower = b[s] < a[s]
        ax.plot([0, 1], [a[s], b[s]], color=COLOR[key_b] if lower else '0.6', lw=0.6, alpha=0.75,
                ls='-' if lower else '--')
    ax.scatter(np.zeros(len(a)), a, marker=MARKER[key_a], s=12, color=COLOR[key_a], zorder=3)
    hollow = key_b == 'clinical_dual_band'
    ax.scatter(np.ones(len(b)), b, marker=MARKER[key_b], s=16, facecolor='white' if hollow else COLOR[key_b],
               edgecolor=COLOR[key_b], zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_a, label_b], fontsize=6.5)
    ax.set_xlim(-0.35, 1.35)


def single_burst(events, duration):
    t = np.arange(0, duration, P.STIM_DT_S)
    return t, burst_forcing(events, t)


def fig1():
    fig = plt.figure(figsize=(7.2, 7.2))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1.05, 1.05], hspace=0.75, wspace=0.28)
    box = [dict(site=0, onset=0.0, dur=T_BURST, env='box', freq=constant(250.0))]
    low = [dict(site=0, onset=0.0, dur=T_BURST, env='gauss', freq=constant(30.0), gain=0.6)]
    high = [dict(site=0, onset=0.0, dur=T_BURST, env='gauss', freq=constant(250.0), gain=0.6)]

    t, f_box = single_burst(box, 0.12)
    _, f_low = single_burst(low, 0.12)
    _, f_high = single_burst(high, 0.12)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[1, 0], sharex=a1, sharey=a1)
    a1.plot(t * 1e3, f_box[1, 0], color=COLOR['regular'], lw=0.5)
    a1.set_title('Regular vCR burst', loc='left')
    a2.plot(t * 1e3, f_high[1, 0], color=COLOR['dual_band'], lw=0.5, label='250 Hz component (FA2)')
    a2.plot(t * 1e3, f_low[0, 0], color=FA1_COLOR, lw=1.2, ls='--', label='30 Hz component (FA1)')
    a2.plot(t * 1e3, 0.6 * np.exp(-((t - T_BURST / 2) ** 2) / (2 * (T_BURST / 4) ** 2)) * (t < T_BURST),
            color='0.4', lw=0.8, ls=':', label='Gaussian envelope')
    a2.set_title('Dual-band burst', loc='left')
    a2.legend(frameon=False, loc='upper right', fontsize=6)
    for ax in (a1, a2):
        ax.set_ylabel('channel input (a.u.)')
        ax.set_ylim(0, 1.15)
    a2.set_xlabel('time from burst onset (ms)')
    letter(a1, 'a', x=-0.24, y=1.08)

    def spectrum(x):
        return np.fft.rfftfreq(len(x), P.STIM_DT_S), np.abs(np.fft.rfft(x)) * P.STIM_DT_S
    t, f_box = single_burst(box, 1.0)
    _, f_low = single_burst(low, 1.0)
    _, f_high = single_burst(high, 1.0)
    b1 = fig.add_subplot(gs[0, 1])
    b2 = fig.add_subplot(gs[1, 1], sharex=b1, sharey=b1)
    f, X = spectrum(f_box.sum(axis=(0, 1)))
    b1.semilogx(f[1:], X[1:], color=COLOR['regular'], lw=0.8)
    f, X = spectrum(f_high[1, 0])
    b2.semilogx(f[1:], X[1:], color=COLOR['dual_band'], lw=0.8, label='250 Hz component')
    f, X = spectrum(f_low[0, 0])
    b2.semilogx(f[1:], X[1:], color=FA1_COLOR, lw=1.0, ls='--', label='30 Hz component')
    for ax in (b1, b2):
        ax.set_xlim(5, 3000)
        ax.set_ylabel('amplitude')
        for x in (60, 500):
            ax.axvline(x, color='0.85', lw=0.6, zorder=0)
    b1.set_title('Spectrum, regular vCR burst', loc='left')
    b2.set_title('Spectrum, dual-band burst', loc='left')
    b2.legend(frameon=False, fontsize=6)
    b2.set_xlabel('frequency (Hz)')
    letter(b1, 'b', x=-0.24, y=1.08)

    def site_timing(ax, schedule, cycles, title, color):
        for e in schedule(np.random.default_rng(11), 0.0, cycles * T_CR):
            ax.add_patch(Rectangle((e['onset'] * 1e3, e['site'] + 0.15), e['dur'] * 1e3, 0.7,
                                   facecolor=color, edgecolor='none', alpha=0.85))
        for c in range(cycles + 1):
            ax.axvline(c * T_CR * 1e3, color='0.6', lw=0.5, ls=':')
        ax.set_xlim(0, cycles * T_CR * 1e3)
        ax.set_ylim(0, 5)
        ax.set_yticks(np.arange(5) + 0.5)
        ax.set_yticklabels([f'site {k + 1}' for k in range(5)], fontsize=6)
        ax.set_title(title, loc='left')
        ax.set_xlabel('time (ms)')

    c1 = fig.add_subplot(gs[2, 0])
    site_timing(c1, regular, 2, 'Staggered timing (regular vCR and dual-band)', COLOR['dual_band'])
    c2 = fig.add_subplot(gs[2, 1])
    site_timing(c2, synchronous, 2, 'Synchronous multi-site control', COLOR['synchronous'])
    letter(c1, 'c', x=-0.24, y=1.08)
    d = fig.add_subplot(gs[3, :])
    site_timing(d, clinical_replica, 5, 'Clinical timing', COLOR['clinical_replica'])
    for c in (3, 4):
        d.axvspan(c * T_CR * 1e3, (c + 1) * T_CR * 1e3, color='0.93', zorder=0)
        d.text((c + 0.5) * T_CR * 1e3, 2.5, 'off cycle', ha='center', va='center', fontsize=7, color='0.4')
    letter(d, 'd', x=-0.06)
    save(fig, 'Fig1_stimulus_patterns')


def fig2():
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 6, hspace=0.75, wspace=1.6)

    ax = fig.add_subplot(gs[0, 0:2])
    f = np.logspace(np.log10(5), np.log10(1000), 400)
    ax.semilogx(f, tuning(f, FA1), color=FA1_COLOR, ls='--', label='FA1')
    ax.semilogx(f, tuning(f, FA2), color=FA2_COLOR, label='FA2')
    ax.set_xlabel('vibration frequency (Hz)')
    ax.set_ylabel('gain $T_c(f)$')
    ax.set_ylim(0, 1.3)
    ax.legend(frameon=False, fontsize=6.5, loc='upper center', ncol=2)
    letter(ax, 'a', x=-0.3)

    ax = fig.add_subplot(gs[0, 2:4])
    x = P.neuron_positions()
    ax.plot(x, receptive_field(FA1.rf_width)[:, 2], color=FA1_COLOR, ls='--', label='FA1 (0.05)')
    ax.plot(x, receptive_field(FA2.rf_width)[:, 2], color=FA2_COLOR, label='FA2 (0.15)')
    top = ax.secondary_xaxis('top')
    top.set_xticks(P.site_positions())
    top.set_xticklabels([str(k + 1) for k in range(P.N_SITES)], fontsize=6)
    top.set_xlabel('site', fontsize=6.5, labelpad=2)
    ax.set_xlabel('position along the network')
    ax.set_ylabel(r'weight $\rho_c(i, 3)$')
    ax.set_ylim(0, 2.2)
    ax.legend(frameon=False, fontsize=6.5, loc='upper left')
    letter(ax, 'b', x=-0.3, y=1.12)

    ax = fig.add_subplot(gs[0, 4:6])
    tt = np.arange(0, T_CR, P.STIM_DT_S)
    forcing = burst_forcing(dual_band(np.random.default_rng(11), 0.0, T_CR), tt)
    drive = np.einsum('cis,cst->it', receptive_fields(ANATOMICAL), forcing)
    ax.imshow(drive[:, ::10], aspect='auto', origin='lower', cmap='Greys', extent=[0, T_CR * 1e3, 0, P.N])
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('neuron')
    letter(ax, 'c', x=-0.3)

    ax = fig.add_subplot(gs[1, 0:3])
    M = mexican_hat(P.N)
    offsets = np.array([o for o in range(-P.N // 2, P.N // 2) if o != 0])
    profile = np.array([M[P.N // 2, (P.N // 2 + o) % P.N] for o in offsets])
    ax.plot(offsets, profile, '.', color='0.2', ms=2.5)
    ax.axhline(0, color='0.7', lw=0.5)
    ax.set_ylim(-0.12, 1.75)
    ax.set_xlabel('neuron distance $j - i$')
    ax.set_ylabel('$M_{ij}$')
    inset = ax.inset_axes([0.72, 0.68, 0.26, 0.30])
    inset.plot(offsets, profile, '.', color='0.2', ms=1.5)
    inset.set_xlim(20, 40)
    inset.set_ylim(-0.06, 0.01)
    inset.axhline(0, color='0.7', lw=0.4)
    inset.tick_params(labelsize=5)
    letter(ax, 'd', x=-0.18)

    ax = fig.add_subplot(gs[1, 3:6])
    dts = np.linspace(-60, 60, 600)
    ax.plot(dts, [stdp_kernel(d, P.BETA1, P.BETA2, P.GAMMA1, P.GAMMA2, P.TAU_STDP) for d in dts], color='0.2')
    ax.axhline(0, color='0.7', lw=0.5)
    ax.axvline(0, color='0.7', lw=0.5)
    ax.set_xlabel(r'$\Delta t = t_{post} - t_{pre}$ (ms)')
    ax.set_ylabel(r'$W(\Delta t)$')
    letter(ax, 'e', x=-0.18)
    for a in fig.axes:
        a.tick_params(labelsize=6.5)
    save(fig, 'Fig2_model')


def traces(condition, sims):
    t, sim, R, Cav = load_traces(RESULTS, condition)
    keep = np.isin(sim, list(sims))
    return t, R[keep], Cav[keep]


def coupling_profile(c, M):
    n = c.shape[0]
    idx = np.arange(n)
    diff = np.abs(idx[:, None] - idx[None, :])
    diff = np.minimum(diff, n - diff)
    return np.array([c[diff == d].mean() for d in range(1, n // 2 + 1)])


def fig3():
    main = data('main.csv')
    vs_all = table('main_dual_band_vs_all.csv').set_index('b')
    confirmation = range(30, 60)
    keys = [('regular', COLOR['regular'], '-'), ('dual_band', COLOR['dual_band'], '-'),
            ('dual_band_shifted', SHIFT_COLOR, '--')]

    fig = plt.figure(figsize=(6.6, 8.2))
    gs = fig.add_gridspec(4, 2, hspace=0.62, wspace=0.45, height_ratios=[1.0, 1.0, 1.0, 1.0])
    rng = np.random.default_rng(1)

    ax = fig.add_subplot(gs[0, :])
    for x, k in enumerate(SHOWN):
        points_with_mean(ax, k, x, by_sim(main, k, 'Cav_post').values, rng)
    ax.set_xticks(range(len(SHOWN)))
    ax.set_xticklabels([SHORT[k] for k in SHOWN], fontsize=6.5)
    baseline = main['Cav_baseline'].mean()
    ax.axhline(baseline, color='0.5', ls='--', lw=0.6)
    ax.text(0.02, baseline + 0.02, 'baseline', fontsize=6, color='0.4', ha='left')
    ax.set_ylabel('$C_{av}$ after\nstimulation')
    ax.set_ylim(-0.30, 0.52)
    letter(ax, 'a', x=-0.08)

    inset = ax.inset_axes([0.30, 0.46, 0.68, 0.50], zorder=6)
    inset.set_facecolor('white')
    for x, k in enumerate(STAGGERED):
        points_with_mean(inset, k, x, by_sim(main, k, 'Cav_post').values, rng, jitter=0.1, size=4,
                         alpha=0.25, mean_size=6, lw=1.4)
    inset.set_xticks(range(len(STAGGERED)))
    inset.set_xticklabels(['Regular', 'Jittered', 'Interleaved', 'Chirped', 'Dual-band'], fontsize=5)
    inset.set_ylim(-0.29, -0.16)
    inset.tick_params(labelsize=5, length=2)
    inset.set_ylabel('$C_{av}$', fontsize=5.5)
    inset.set_title('staggered patterns, expanded scale', fontsize=5.5, color='0.3', pad=2)

    ax = fig.add_subplot(gs[1:3, 0])
    order = ['regular', 'jittered', 'interleaved', 'chirped'] + OTHER
    for y, k in enumerate(order):
        r = vs_all.loc[k]
        ax.errorbar(r.dz, y, xerr=[[r.dz - r.ci_low], [r.ci_high - r.dz]], fmt=MARKER[k], color=COLOR[k],
                    ms=4.5, capsize=2, lw=1)
        ax.text(1.03, y, f'{int(r.a_lower)}/{int(r.n)}', transform=ax.get_yaxis_transform(),
                fontsize=6, va='center')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABEL[k] for k in order], fontsize=6.5)
    ax.invert_yaxis()
    ax.axvline(0, color='0.5', lw=0.6)
    ax.set_xlabel('$d_z$ (95% CI), dual-band\nminus pattern')
    letter(ax, 'b', x=-0.62)

    axR = fig.add_subplot(gs[1, 1])
    axC = fig.add_subplot(gs[2, 1])
    labels = {'regular': 'Regular vCR', 'dual_band': 'Dual-band CR',
              'dual_band_shifted': 'Dual-band, 90° shift'}
    for k, color, ls in keys:
        t, R, C = traces(k, confirmation)
        for ax, Y in ((axR, R), (axC, C)):
            q1, med, q3 = np.percentile(Y, [25, 50, 75], axis=0)
            ax.fill_between(t, q1, q3, color=color, alpha=0.2, lw=0)
            ax.plot(t, med, color=color, lw=1.0, ls=ls, label=labels[k])
    for ax in (axR, axC):
        ax.axvspan(0, 128, color='0.93', zorder=0)
        ax.set_xlim(-60, 256)
        ax.set_xlabel('time from stimulation onset (s)')
    axR.set_ylabel('order parameter R')
    axR.set_ylim(0, 1.0)
    axR.text(64, 0.90, 'stimulation', ha='center', fontsize=6, color='0.4')
    axC.set_ylabel('$C_{av}$')
    axC.legend(frameon=False, fontsize=5.5, loc='center left', bbox_to_anchor=(0.01, 0.30),
               handlelength=1.6)
    letter(axR, 'c', x=-0.3)
    letter(axC, 'd', x=-0.3)

    ax = fig.add_subplot(gs[3, 0])
    for k, color, ls in keys:
        t, _, C = traces(k, confirmation)
        m = t > 120
        q1, med, q3 = np.percentile(C[:, m], [25, 50, 75], axis=0)
        ax.fill_between(t[m], q1, q3, color=color, alpha=0.2, lw=0)
        ax.plot(t[m], med, color=color, lw=1.0, ls=ls)
    lo, _ = coupling_bounds(mexican_hat(P.N))
    ax.axhline(lo, color='0.5', lw=0.6, ls=':')
    ax.text(254, lo + 0.002, 'lower bound', fontsize=6, color='0.4', ha='right', va='bottom')
    ax.axvline(128, color='0.6', lw=0.6, ls='--')
    ax.set_xlim(120, 256)
    ax.set_xlabel('time from stimulation onset (s)')
    ax.set_ylabel('$C_{av}$')
    letter(ax, 'e', x=-0.3)

    ax = fig.add_subplot(gs[3, 1])
    M = mexican_hat(P.N)
    for k, color, ls in keys:
        path = os.path.join(RESULTS, 'example', f'example_{k}.npz')
        if not os.path.exists(path):
            continue
        d = np.load(path)
        ax.plot(np.arange(1, P.N // 2 + 1), coupling_profile(d['c_stim'], M), color=color, lw=1.0, ls=ls)
        ax.plot(np.arange(1, P.N // 2 + 1), coupling_profile(d['c_post'], M), color=color, lw=1.0, ls=ls,
                alpha=0.45)
    ax.axvline(27.5, color='0.7', lw=0.6, ls=':')
    ax.text(26.5, 0.30, 'excitatory', ha='right', fontsize=6, color='0.4')
    ax.text(28.5, 0.30, 'inhibitory', ha='left', fontsize=6, color='0.4')
    ax.set_xlabel('ring distance $|i - j|$')
    ax.set_ylabel('mean weight $c_{ij}$')
    ax.set_ylim(-0.03, 1.03)
    ax.plot([], [], color='0.3', lw=1.0, label='end of stimulation')
    ax.plot([], [], color='0.3', lw=1.0, alpha=0.45, label='end of post period')
    ax.legend(frameon=False, fontsize=5.5, loc='center left', bbox_to_anchor=(0.20, 0.52),
              handlelength=1.6)
    letter(ax, 'f', x=-0.3)
    save(fig, 'Fig3_main_comparison')


CONTROL_ROWS = [
    ('Dose', 'carrier_at_dual_power', 'Carrier at dual-band power'),
    ('Dose', 'carrier_at_dual_drive', 'Carrier at dual-band mean drive'),
    ('Envelope', 'gauss250_at_dual_power', '250 Hz Gaussian, matched power'),
    ('Envelope', 'gauss250_at_dual_drive', '250 Hz Gaussian, matched mean drive'),
    ('Single component', 'carrier_only', '250 Hz carrier'),
    ('Single component', 'only_30hz', '30 Hz only'),
    ('Component identity', 'constant_substitution', 'Constant substitution'),
    ('Component identity', 'two_field_250hz', '250 Hz, two receptive fields'),
    ('Receptive field', 'both_focal', 'Both components focal'),
    ('Receptive field', 'both_broad', 'Both components broad'),
    ('30 Hz phase', 'dual_band_phase45', 'Shifted by 45°'),
    ('30 Hz phase', 'dual_band_shifted', 'Shifted by 90°'),
    ('30 Hz phase', 'dual_band_phase135', 'Shifted by 135°'),
    ('30 Hz phase', 'dual_band_random_phase', 'Random phase'),
]


def fig4():
    ctrl = table('controls.csv').set_index('b')
    phase = table('phase_tests.csv')
    phase = phase[(phase['sims'] == '0-29') & (phase['b'] == 'dual_band') & (phase['metric'] == 'Cav_post')]
    phase = phase.set_index('a')
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    fig.subplots_adjust(left=0.40, right=0.74)
    y, ticks, labels, last = 0, [], [], None
    for group, cond, label in CONTROL_ROWS:
        if cond in ctrl.index:
            r = ctrl.loc[cond]
            dz, lo, hi, p, lower, n, sig = r.dz, r.ci_low, r.ci_high, r.p, r.a_lower, r.n, bool(r.holm)
        elif cond in phase.index:
            r = phase.loc[cond]       # phase variant minus dual-band; flip to dual-band minus variant
            dz, lo, hi, p, lower, n = -r.dz, -r.ci_high, -r.ci_low, r.p, r.n - r.a_lower, r.n
            sig = p < 0.05
        else:
            continue
        y += 1.0 if group != last and last else 0
        if group != last:
            ax.text(-0.02, y + 0.35, group, fontsize=7, fontweight='bold', va='center', ha='right',
                    transform=ax.get_yaxis_transform())
            y += 0.75
            last = group
        else:
            y += 0.75
        color = COLOR['dual_band'] if sig else '0.45'
        ax.errorbar(dz, y, xerr=[[dz - lo], [hi - dz]], fmt='o', color=color, mfc=color if sig else 'white',
                    ms=4.5, capsize=2, lw=1)
        ax.text(1.03, y, f'n = {int(n)}, {int(lower)} lower;  {format_p(p)}', transform=ax.get_yaxis_transform(),
                fontsize=6.2, va='center')
        ticks.append(y)
        labels.append(label)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.6)
    ax.invert_yaxis()
    ax.axvline(0, color='0.5', lw=0.7)
    ax.set_xlabel('$d_z$ (95% CI), dual-band burst minus control')
    save(fig, 'Fig4_controls')


def fig5():
    sn = data('single_neuron.csv')
    tr = np.load(os.path.join(RESULTS, 'single_neuron_traces.npz'))
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.05], hspace=0.75, wspace=0.55)

    t = tr['t_ms']
    td = np.arange(len(tr['D_regular'])) * P.STIM_DT_MS
    for col, (key, title, color) in enumerate([('regular', 'Regular vCR burst', COLOR['regular']),
                                               ('phase0', 'Dual-band, phase-locked', COLOR['dual_band']),
                                               ('phase90', 'Dual-band, shifted by 90°', SHIFT_COLOR)]):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(t - 300, tr[f'V_{key}'], color='0.15', lw=0.5)
        ax.set_xlim(-60, 260)
        ax.set_ylim(-85, 60)
        ax.set_title(title, fontsize=7, loc='left')
        ax.set_xlabel('time from burst onset (ms)')
        if col == 0:
            ax.set_ylabel('V (mV)')
            letter(ax, 'a', x=-0.4, y=1.12)
        drive = ax.twinx()
        drive.fill_between(td - 300, 0, tr[f'D_{key}'], color=color, alpha=0.35, lw=0)
        drive.set_ylim(0, 4.0)
        drive.set_yticks([])
        drive.spines['right'].set_visible(False)

    ax = fig.add_subplot(gs[1, 0:2])
    ph = sn[(sn['burst'] == 'dual_band') & (sn.get('sweep').isna() if 'sweep' in sn else True)]
    g = ph.groupby('phase')['longest_interval_ms']
    x = np.array(sorted(ph['phase'].unique()))
    lo, med, hi = (np.append(v.loc[x].values, v.loc[x[0]]) for v in (g.min(), g.median(), g.max()))
    x = np.append(x, 180)   # the rectified component has a period of 180 degrees
    ax.fill_between(x, lo, hi, color=COLOR['dual_band'], alpha=0.2, lw=0)
    ax.plot(x, med, 'o-', color=COLOR['dual_band'], ms=3, lw=1)
    reg = sn[(sn['burst'] == 'regular') & sn['sweep'].isna()]['longest_interval_ms']
    ax.axhspan(reg.min(), reg.max(), color=COLOR['regular'], alpha=0.2, lw=0)
    ax.text(105, reg.max() + 3, 'regular vCR burst', fontsize=6, color=COLOR['regular'], ha='center', va='bottom')
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_xlim(-5, 185)
    ax.set_xlabel('phase of the 30 Hz component (degrees)')
    ax.set_ylabel('longest interval after\nburst onset (ms)')
    letter(ax, 'b', x=-0.16)

    ax = fig.add_subplot(gs[1, 2])
    cs = sn[sn['sweep'] == 'current']
    for burst, label in (('dual_band', 'dual-band'), ('regular', 'regular')):
        color = COLOR[burst]
        d = cs[cs['burst'] == burst].sort_values('current')
        ax.plot(d['current'], d['longest_interval_ms'], 'o-', color=color, ms=2.5, lw=1, label=label)
    ax.axvspan(P.I0 - P.DELTA_I, P.I0 + P.DELTA_I, color='0.9', zorder=0)
    ax.set_xlabel(r'input current $I_i$ ($\mu$A/cm$^2$)')
    ax.set_ylabel('longest interval (ms)')
    ax.set_ylim(0, 270)
    ax.legend(frameon=False, fontsize=6, loc='upper right')
    letter(ax, 'c', x=-0.45)

    ax = fig.add_subplot(gs[2, 0:2])
    main = data('main.csv')
    sweep = os.path.join(RESULTS, 'phase_sweep.csv')
    ph = pd.concat([data('phase.csv')] + ([data('phase_sweep.csv')] if os.path.exists(sweep) else []))
    ph = ph[ph['sim'] < 30]
    rows = [('0°', by_sim(main, 'dual_band', 'Cav_post'), COLOR['dual_band'])]
    for cond, label in (('dual_band_phase45', '45°'), ('dual_band_shifted', '90°'), ('dual_band_phase135', '135°'),
                        ('dual_band_random_phase', 'random')):
        if cond in set(ph['condition']):
            rows.append((label, by_sim(ph, cond, 'Cav_post'), SHIFT_COLOR))
    carrier = by_sim(main, 'regular', 'Cav_post')
    m, lo, hi = ci95(carrier)
    ax.axhspan(lo, hi, color=COLOR['regular'], alpha=0.2, lw=0)
    ax.axhline(m, color=COLOR['regular'], lw=0.8, label='regular vCR (mean, 95% CI)')
    ax.legend(frameon=False, fontsize=6, loc='upper left')
    rng = np.random.default_rng(4)
    for x, (label, values, color) in enumerate(rows):
        ax.scatter(x + rng.uniform(-0.1, 0.1, len(values)), values, s=5, color=color, alpha=0.3, lw=0)
        mm, l, h = ci95(values)
        ax.errorbar(x + 0.25, mm, yerr=[[mm - l], [h - mm]], fmt='o', color=color, ms=5, capsize=3, lw=1.2,
                    mec='black', mew=0.5)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_xlim(-0.5, len(rows) - 0.4)
    ax.set_xlabel('phase of the 30 Hz component')
    ax.set_ylabel('$C_{av}$ after stimulation')
    letter(ax, 'd', x=-0.16)

    ax = fig.add_subplot(gs[2, 2])
    path = os.path.join(TABLES, 'mechanism.csv')
    if os.path.exists(path):
        mech = pd.read_csv(path).set_index('condition')
        x = mech['long_intervals_mean'] / 20.0 / P.N
        y = mechanism_outcome(mech.index)
        y = y[y < -0.15]
        for k in mech.index:
            if k in y.index:
                ax.scatter(x[k], y[k], s=14, marker=MECH_MARKER.get(k, 'o'), color=MECH_COLOR.get(k, '0.5'),
                           edgecolor='black', linewidth=0.3, zorder=3)
        ax.set_xscale('symlog', linthresh=0.01)
        ax.set_ylim(-0.27, -0.155)
        ax.set_xlabel('pauses per neuron per s')
        ax.set_ylabel('$C_{av}$ after stimulation')
    letter(ax, 'e', x=-0.45)
    save(fig, 'Fig5_pauses')


def fig6():
    mech = table('mechanism.csv').set_index('condition')
    y = mechanism_outcome(mech.index)
    y = y[y < -0.15]
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.6)

    ax = fig.add_subplot(gs[0, 0:2])
    for k in mech.index:
        ax.scatter(mech.loc[k, 'sector_input_coherence'], mech.loc[k, 'sector_coherence_mean'], s=18,
                   marker=MECH_MARKER.get(k, 'o'), color=MECH_COLOR.get(k, '0.5'), edgecolor='black',
                   linewidth=0.3, zorder=3)
    ax.plot([0, 1], [0, 1], color='0.7', lw=0.6, ls=':')
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel('sector input coherence')
    ax.set_ylabel('network sector coherence\nduring stimulation')
    letter(ax, 'a', x=-0.16)
    handles = [plt.Line2D([], [], ls='', marker=MECH_MARKER[k], color=MECH_COLOR[k], mec='black', mew=0.3,
                          label=MECH_LABEL[k]) for k in MECH_LEGEND]
    ax.legend(handles=handles, frameon=False, fontsize=5.8, loc='center left', bbox_to_anchor=(1.02, 0.5))

    ax = fig.add_subplot(gs[1, 0])
    for k in mech.index:
        if k in y.index:
            ax.scatter(mech.loc[k, 'sector_coherence_mean'], y[k], s=14, marker=MECH_MARKER.get(k, 'o'),
                       color=MECH_COLOR.get(k, '0.5'), edgecolor='black', linewidth=0.3, zorder=3)
    ax.set_xlabel('network sector coherence\nduring stimulation')
    ax.set_ylabel('$C_{av}$ after stimulation')
    ax.set_ylim(-0.27, -0.155)
    letter(ax, 'b', x=-0.42)

    coh = data('coherence.csv')
    stim = coh[coh['window'] == 'stimulation']
    for j, (metric, ylabel, l) in enumerate([('sector_coherence', 'sector coherence\nduring stimulation', 'c'),
                                             ('sync_band_power', 'synchrony-band power\nduring stimulation', 'd')]):
        ax = fig.add_subplot(gs[1, 1 + j])
        paired_lines(ax, by_sim(stim, 'regular', metric), by_sim(stim, 'dual_band', metric),
                     'regular', 'dual_band', 'Regular\nvCR', 'Dual-\nband')
        ax.set_ylabel(ylabel)
        letter(ax, l, x=-0.55)
    save(fig, 'Fig6_selectivity')


CLINICAL_FILE, CLINICAL_PAIR = 'clinical.csv', ('clinical_replica', 'clinical_dual_band')


def fig7():
    clin = data(CLINICAL_FILE)
    a, b = CLINICAL_PAIR
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 4, hspace=0.65, wspace=0.9)
    metrics = [('Cav_post', '$C_{av}$ after stimulation'),
               ('Cav_weighted_post', '|M|-weighted coupling\nafter stimulation'),
               ('R_post', 'R after stimulation'), ('R_on', 'R during stimulation')]
    for j, ((m, ylabel), l) in enumerate(zip(metrics, 'abcd')):
        ax = fig.add_subplot(gs[0, j])
        paired_lines(ax, by_sim(clin, a, m), by_sim(clin, b, m), 'clinical_replica', 'clinical_dual_band',
                     'Replica', 'Dual-\nband')
        ax.set_ylabel(ylabel)
        letter(ax, l, x=-0.75)

    ax = fig.add_subplot(gs[1, 0:2])
    main = data('main.csv')
    curves = [(by_sim(clin, a, 'Cav_post'), COLOR['clinical_replica'], '-', 'Clinical-style replica'),
              (by_sim(clin, b, 'Cav_post'), COLOR['clinical_dual_band'], '--', 'Clinical-style dual-band'),
              (by_sim(main, 'regular', 'Cav_post'), COLOR['regular'], ':', 'Regular vCR, continuous')]
    grid = np.linspace(-0.30, -0.14, 400)
    for values, color, ls, label in curves:
        ax.plot(grid, [(values < g).mean() for g in grid], color=color, ls=ls, lw=1.2, label=label)
    ax.axvline(-0.24, color='0.7', lw=0.6, ls=':')
    ax.set_xlabel('coupling level $C^*$')
    ax.set_ylabel('fraction of networks\nbelow $C^*$')
    ax.set_xlim(-0.30, -0.14)
    ax.set_xticks(np.arange(-0.30, -0.139, 0.04))
    ax.legend(frameon=False, fontsize=6.2, loc='center left', bbox_to_anchor=(1.05, 0.5))
    letter(ax, 'e', x=-0.22, y=1.06)
    save(fig, 'Fig7_clinical_protocol')


ROBUST_NAMES = {'reference': 'Reference model', 'K_stim 0.20': '$K_{stim}$ 0.20', 'K_stim 0.50': '$K_{stim}$ 0.50',
                'plasticity rate x0.5': 'Plasticity rate x0.5', 'plasticity rate x2': 'Plasticity rate x2',
                'a2 1.5': 'Mexican-hat width 1.5', 'a2 2.5': 'Mexican-hat width 2.5',
                'bandwidth 1.0': 'Tuning bandwidth 1.0 octave', 'bandwidth 1.5': 'Tuning bandwidth 1.5 octaves',
                'input current 14': r'Input current 14 $\mu$A/cm$^2$', '200 neurons': '200 neurons'}


def fig8():
    rob = table('robustness.csv')
    rob = rob[rob['metric'] == 'Cav_post'].set_index('change')
    size = table('network_size_tests.csv').iloc[0]
    fig = plt.figure(figsize=(6.4, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.9, 1], hspace=0.6, wspace=1.1)

    ax = fig.add_subplot(gs[:, 0])
    rows = []
    for change, label in ROBUST_NAMES.items():
        if change == '200 neurons':
            r, sig = size, size.p < 0.05
        elif change in rob.index:
            r = rob.loc[change]
            sig = change == 'reference' or bool(r.get('holm', False))
        else:
            continue
        rows.append((label, r.dz, r.ci_low, r.ci_high, r.p, f'{int(r.a_lower)}/{int(r.n)}', sig))
    for y, (label, d, lo, hi, p, count, sig) in enumerate(rows):
        color = COLOR['dual_band'] if sig else '0.45'
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt='s' if y == 0 else 'o', color=color,
                    mfc=color if sig else 'white', ms=4.5, capsize=2, lw=1)
        ax.text(1.03, y, f'{count}, {format_p(p)}', transform=ax.get_yaxis_transform(), fontsize=6, va='center')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax.invert_yaxis()
    ax.axvline(0, color='0.5', lw=0.6)
    ax.set_xlabel('$d_z$ (95% CI), dual-band\nminus regular vCR')
    letter(ax, 'a', x=-0.95)

    surface = table('gain_sweep_surface.csv').set_index('gain')
    ax = fig.add_subplot(gs[0, 1])
    values = surface.values
    im = ax.imshow(values, cmap='viridis_r', aspect='auto', origin='lower')
    ax.set_xticks(range(values.shape[1]))
    ax.set_xticklabels([f'{float(c):.0f}' for c in surface.columns])
    ax.set_yticks(range(values.shape[0]))
    ax.set_yticklabels([f'{g:.1f}' for g in surface.index])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f'{values[i, j]:.3f}', ha='center', va='center', fontsize=5.2,
                    color='black' if values[i, j] < -0.245 else 'white')
    ax.set_xlabel('second-component frequency (Hz)')
    ax.set_ylabel('component amplitude')
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    cb.ax.tick_params(labelsize=5.5)
    cb.set_label('$C_{av}$ after stimulation', fontsize=6)
    letter(ax, 'b', x=-0.4)

    cal = table('calibration.csv')
    ax = fig.add_subplot(gs[1, 1])
    for k in ('synchronous', 'regular', 'dual_band'):
        d = cal[cal['condition'] == k].sort_values('K_stim')
        ax.errorbar(d['K_stim'], d['R_on'], yerr=d['R_on_sd'], color=COLOR[k], marker=MARKER[k], ms=3.5, lw=1,
                    ls='-' if k == 'dual_band' else '--', capsize=1.5, label=SHORT[k].replace('\n', ' '))
    ax.axvline(P.K_STIM, color='0.6', lw=0.6, ls=':')
    ax.axhline(cal['R_baseline'].mean(), color='0.7', lw=0.6, ls='--')
    ax.set_xlabel('$K_{stim}$ (mS/cm$^2$)')
    ax.set_ylabel('R during stimulation')
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=5.8, loc='center right', bbox_to_anchor=(1.0, 0.55))
    letter(ax, 'c', x=-0.4)
    save(fig, 'Fig8_robustness')


def figS4():
    z = np.load(os.path.join(RESULTS, 'coherence_spectra.npz'))
    freq, power = z['freq'], z['power']
    index = {}
    for key, p in zip(z['keys'], power):
        cond, _, window = key.split('|')
        index.setdefault((cond, window), []).append(p)
    keys = ['synchronous', 'regular', 'dual_band', 'clinical_replica', 'clinical_dual_band']
    names = {'synchronous': 'Synchronous control', 'regular': 'Regular vCR', 'dual_band': 'Dual-band CR',
             'clinical_replica': 'Clinical-style replica', 'clinical_dual_band': 'Clinical-style dual-band'}
    band = (freq >= 30) & (freq <= 110)
    windows = [('baseline', 'Baseline'), ('stimulation', 'Stimulation'), ('post', 'After stimulation')]
    fig, axes = plt.subplots(len(keys), 3, figsize=(7.2, 7.0), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0.3, wspace=0.1))
    for i, k in enumerate(keys):
        for j, (w, title) in enumerate(windows):
            ax = axes[i, j]
            arr = np.array(index[(k, w)])
            q1, med, q3 = np.percentile(arr, [25, 50, 75], axis=0)
            ax.fill_between(freq[band], q1[band], q3[band], color=COLOR[k], alpha=0.25, lw=0)
            ax.semilogy(freq[band], med[band], color=COLOR[k], lw=0.9)
            ax.axvspan(55, 85, color='0.94', zorder=0)
            if i == 0:
                ax.set_title(title, fontsize=7)
            if j == 0:
                ax.set_ylabel(f'{names[k]}\npower (counts$^2$/Hz)', fontsize=6)
            if i == len(keys) - 1:
                ax.set_xlabel('frequency (Hz)')
            ax.tick_params(labelsize=5.5)
    save(fig, 'FigS4_spectra')


def figS1():
    main = data('main.csv')
    order = SHOWN + OTHER
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.0), sharex=True, gridspec_kw=dict(hspace=0.25))
    rng = np.random.default_rng(5)
    for ax, (m, ylabel) in zip(axes, [('R_on', 'R during stimulation'), ('R_post', 'R after stimulation'),
                                      ('Cav_post', '$C_{av}$ after stimulation'),
                                      ('Cav_weighted_post', '|M|-weighted coupling\nafter stimulation')]):
        for x, k in enumerate(order):
            points_with_mean(ax, k, x, by_sim(main, k, m).values, rng)
        ax.set_ylabel(ylabel)
    axes[-1].set_xticks(range(len(order)))
    axes[-1].set_xticklabels([SHORT[k] for k in order], fontsize=6)
    save(fig, 'FigS1_all_patterns')


EXAMPLES = ['synchronous', 'regular', 'dual_band', 'dual_band_shifted', 'smooth']


def example(k):
    return np.load(os.path.join(RESULTS, 'example', f'example_{k}.npz'))


def figS2():
    fig, axes = plt.subplots(len(EXAMPLES), 3, figsize=(7.2, 8.4), sharey=True,
                             gridspec_kw=dict(hspace=0.5, wspace=0.08))
    cycle = T_CR * 1000
    for r, k in enumerate(EXAMPLES):
        d = example(k)
        t, i = d['spike_t'], d['spike_i']
        on, off = float(d['stim_on_ms']), float(d['stim_off_ms'])
        for c, (t0, title) in enumerate([(on - cycle, 'Baseline'), (on + 1000, 'Stimulation'),
                                         (off + 1000, 'After stimulation')]):
            ax = axes[r, c]
            m = (t >= t0) & (t < t0 + cycle)
            ax.scatter(t[m] - t0, i[m], s=1.2, color='0.1', marker='|', linewidths=0.4, rasterized=True)
            ax.set_xlim(0, cycle)
            ax.set_ylim(-1, P.N)
            if r == 0:
                ax.set_title(title, fontsize=7.5)
            if c == 0:
                ax.set_ylabel(f'{LABEL[k]}\nneuron', fontsize=6.5)
            if r == len(EXAMPLES) - 1:
                ax.set_xlabel('time (ms)')
            ax.tick_params(labelsize=6)
    save(fig, 'FigS2_rasters')


def figS3():
    fig, axes = plt.subplots(len(EXAMPLES), 3, figsize=(6.2, 9.2), gridspec_kw=dict(hspace=0.4, wspace=0.1))
    for r, k in enumerate(EXAMPLES):
        d = example(k)
        for c, (key, title) in enumerate([('c_baseline', 'End of baseline'), ('c_stim', 'End of stimulation'),
                                          ('c_post', 'End of post-stimulation')]):
            ax = axes[r, c]
            im = ax.imshow(d[key], cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([0, 40, 79])
            ax.set_yticks([0, 40, 79])
            ax.tick_params(labelsize=5.5)
            if r == 0:
                ax.set_title(title, fontsize=7.5)
            if c == 0:
                ax.set_ylabel(f'{LABEL[k]}\npostsynaptic neuron', fontsize=6.5)
            else:
                ax.set_yticklabels([])
            if r == len(EXAMPLES) - 1:
                ax.set_xlabel('presynaptic neuron', fontsize=6.5)
            else:
                ax.set_xticklabels([])
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label('$c_{ij}$', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    save(fig, 'FigS3_weight_matrices')


FIGURES_BY_NAME = {'1': fig1, '2': fig2, '3': fig3, '4': fig4, '5': fig5, '6': fig6, '7': fig7, '8': fig8,
                   'S1': figS1, 'S2': figS2, 'S3': figS3, 'S4': figS4}

if __name__ == '__main__':
    for name in sys.argv[1:] or FIGURES_BY_NAME:
        FIGURES_BY_NAME[name]()
