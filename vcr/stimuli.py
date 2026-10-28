"""Stimulation schedules and the named conditions compared in the study.

A schedule is a list of burst events: dicts with site, onset, dur, env and freq,
and optionally gain and phase0. Times are in seconds.

The order of the condition lists at the end of this file matters, because the
random part of a schedule is seeded with the position of the condition in the
list that a study runs.
"""
import numpy as np

from .afferents import ANATOMICAL, FA1, FA2, FOCAL_250, BROAD_30
from .params import N_SITES

T_CR = 1 / 1.5
SLOT = T_CR / N_SITES
T_BURST = 0.1
F_CARRIER = 250.0
F_LOW = 30.0
JITTER = 0.235
FOUR_SITES = 4
CYCLES_ON, CYCLES_OFF = 3, 2

LOGISTIC_R = 3.9
CYCLE_RANGE = (0.4, 0.9)
CARRIER_RANGE = (100.0, 300.0)
CARRIER_PERIODS_PER_BURST = 10
CHIRP_RANGE = (180.0, 320.0)

DUAL_GAIN = 0.6
DUAL_GAIN_POWER_MATCHED = 1.064519        # gain at which the dual-band burst has the carrier's power

_tt = np.linspace(0.0, T_BURST, 10001)
GAUSS_ENVELOPE_MEAN = float(np.mean(np.exp(-((_tt - T_BURST / 2) ** 2) / (2 * (T_BURST / 4) ** 2))))

CARRIER_GAIN_AT_DUAL_POWER = DUAL_GAIN / DUAL_GAIN_POWER_MATCHED
CARRIER_GAIN_AT_DUAL_DRIVE = 2 * DUAL_GAIN * GAUSS_ENVELOPE_MEAN
GAUSS250_GAIN_AT_DUAL_POWER = DUAL_GAIN * np.sqrt(2)
GAUSS250_GAIN_AT_DUAL_DRIVE = 2 * DUAL_GAIN
TWO_FIELD_GAIN = 0.707107
CONSTANT_GAIN = DUAL_GAIN * 2 / np.pi     # the mean of |sin| is 2 / pi


def constant(f):
    return lambda tt: np.full_like(tt, f)


def staggered(rng, t0, t1, burst, jitter=False, duty_cycle=False, n_sites=N_SITES, jitter_range=None):
    """New random site order in every CR cycle; burst(site, onset) returns the events of one burst."""
    slot = T_CR / n_sites
    jitter_range = JITTER * slot if jitter_range is None else jitter_range
    events, t, cycle = [], t0, 0
    while t < t1:
        if not duty_cycle or cycle % (CYCLES_ON + CYCLES_OFF) < CYCLES_ON:
            for j, site in enumerate(rng.permutation(n_sites)):
                onset = t + j * slot
                if jitter:
                    onset += rng.uniform(-jitter_range, jitter_range)
                events.extend(burst(site, onset))
        t += T_CR
        cycle += 1
    return events


def with_burst(burst):
    return lambda rng, t0, t1: staggered(rng, t0, t1, burst)


def on_four_sites(burst, clinical=False):
    if clinical:
        return lambda rng, t0, t1: staggered(rng, t0, t1, burst, jitter=True, duty_cycle=True,
                                             n_sites=FOUR_SITES,
                                             jitter_range=JITTER * T_CR / (2 * FOUR_SITES))
    return lambda rng, t0, t1: staggered(rng, t0, t1, burst, n_sites=FOUR_SITES)


def box_burst(freq=F_CARRIER, gain=None):
    def burst(site, onset):
        ev = dict(site=site, onset=onset, dur=T_BURST, env='box', freq=constant(freq))
        if gain is not None:
            ev['gain'] = gain
        return [ev]
    return burst


def gauss_burst(freq=F_CARRIER, gain=1.0):
    def burst(site, onset):
        return [dict(site=site, onset=onset, dur=T_BURST, env='gauss', freq=constant(freq), gain=gain)]
    return burst


def dual_band_burst(gain=DUAL_GAIN, f1=F_LOW, f2=F_CARRIER, phase=0.0, phase_rng=None):
    """Two components under one Gaussian envelope.

    `phase` shifts the first component against the envelope; with `phase_rng` the
    shift is drawn uniformly from [0, 2 pi) for every burst.
    """
    def burst(site, onset):
        phase0 = phase if phase_rng is None else phase_rng.uniform(0.0, 2 * np.pi)
        return [dict(site=site, onset=onset, dur=T_BURST, env='gauss', freq=constant(f1), gain=gain, phase0=phase0),
                dict(site=site, onset=onset, dur=T_BURST, env='gauss', freq=constant(f2), gain=gain)]
    return burst


def constant_substitution_burst(site, onset):
    return [dict(site=site, onset=onset, dur=T_BURST, env='gauss', freq=constant(F_LOW), gain=DUAL_GAIN),
            dict(site=site, onset=onset, dur=T_BURST, env='dc_gauss', freq=constant(F_CARRIER), gain=CONSTANT_GAIN)]


def chirp_burst(site, onset):
    lo, hi = CHIRP_RANGE

    def freq(tt):
        return lo + (hi - lo) * np.clip(tt / T_BURST, 0, 1)
    return [dict(site=site, onset=onset, dur=T_BURST, env='box', freq=freq, gain=1.0)]


def synchronous(rng, t0, t1):
    events, t = [], t0
    while t < t1:
        for k in range(N_SITES):
            events.append(dict(site=k, onset=t, dur=T_BURST, env='box', freq=constant(F_CARRIER)))
        t += T_CR
    return events


def regular(rng, t0, t1):
    return staggered(rng, t0, t1, box_burst())


def jittered(rng, t0, t1):
    events, t = [], t0
    while t < t1:
        for j, site in enumerate(rng.permutation(N_SITES)):
            jitter = rng.uniform(-JITTER * SLOT, JITTER * SLOT)
            events.append(dict(site=site, onset=t + j * SLOT + jitter, dur=T_BURST, env='box',
                               freq=constant(F_CARRIER)))
        t += T_CR
    return events


def on_off(rng, t0, t1):
    return staggered(rng, t0, t1, box_burst(), duty_cycle=True)


def smooth(rng, t0, t1):
    """Continuous 30 Hz sinusoid at each site, phase-shifted between sites and rectified
    like every other stimulus, so that the drive stays non-negative."""
    return [dict(site=k, onset=t0, dur=t1 - t0, env='rsine', freq=constant(F_LOW),
                 phase0=2 * np.pi * k / N_SITES) for k in range(N_SITES)]


def chaotic_timing(rng, t0, t1):
    events, t = [], t0
    u, v = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)
    while t < t1:
        u = LOGISTIC_R * u * (1 - u)
        v = LOGISTIC_R * v * (1 - v)
        cycle = CYCLE_RANGE[0] + (CYCLE_RANGE[1] - CYCLE_RANGE[0]) * u
        freq = CARRIER_RANGE[0] + (CARRIER_RANGE[1] - CARRIER_RANGE[0]) * v
        dur = CARRIER_PERIODS_PER_BURST / freq
        spacing = cycle / N_SITES
        for j, site in enumerate(rng.permutation(N_SITES)):
            events.append(dict(site=site, onset=t + j * spacing, dur=dur, env='gauss', freq=constant(freq)))
        t += cycle
    return events


def chaotic_order(rng, t0, t1):
    events, t = [], t0
    u = rng.uniform(0.05, 0.95)
    while t < t1:
        keys = []
        for _ in range(N_SITES):
            u = LOGISTIC_R * u * (1 - u)
            keys.append(u)
        for j, site in enumerate(np.argsort(keys)):
            events.append(dict(site=site, onset=t + j * SLOT, dur=T_BURST, env='box', freq=constant(F_CARRIER)))
        t += T_CR
    return events


def interleaved(rng, t0, t1):
    events, t, cycle = [], t0, 0
    while t < t1:
        freq = F_LOW if cycle % 2 == 0 else F_CARRIER
        for j, site in enumerate(rng.permutation(N_SITES)):
            events.append(dict(site=site, onset=t + j * SLOT, dur=T_BURST, env='box', freq=constant(freq)))
        t += T_CR
        cycle += 1
    return events


def chirped(rng, t0, t1):
    return staggered(rng, t0, t1, chirp_burst)


def dual_band(rng, t0, t1, gain=DUAL_GAIN, f2=F_CARRIER):
    return staggered(rng, t0, t1, dual_band_burst(gain=gain, f2=f2))


def dual_band_phase(degrees):
    return lambda rng, t0, t1: staggered(rng, t0, t1, dual_band_burst(phase=np.radians(degrees)))


def dual_band_random_phase(rng, t0, t1):
    """The phases come from a generator spawned from `rng`, so that the site order and
    the burst onsets stay those of dual_band with the same seed."""
    phase_rng = np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0])
    return staggered(rng, t0, t1, dual_band_burst(phase_rng=phase_rng))


def clinical_replica(rng, t0, t1):
    return staggered(rng, t0, t1, box_burst(), jitter=True, duty_cycle=True)


def clinical_dual_band(rng, t0, t1):
    return staggered(rng, t0, t1, dual_band_burst(), jitter=True, duty_cycle=True)


# name: (schedule, afferent channels, closed-loop gating)
CONDITIONS = {
    'synchronous': (synchronous, ANATOMICAL, False),
    'regular': (regular, ANATOMICAL, False),
    'jittered': (jittered, ANATOMICAL, False),
    'on_off': (on_off, ANATOMICAL, False),
    'smooth': (smooth, ANATOMICAL, False),
    'chaotic_timing': (chaotic_timing, ANATOMICAL, False),
    'chaotic_order': (chaotic_order, ANATOMICAL, False),
    'interleaved': (interleaved, ANATOMICAL, False),
    'chirped': (chirped, ANATOMICAL, False),
    'dual_band': (dual_band, ANATOMICAL, False),
    'closed_loop': (regular, ANATOMICAL, True),
    'carrier_at_dual_power': (with_burst(box_burst(gain=CARRIER_GAIN_AT_DUAL_POWER)), ANATOMICAL, False),
    'carrier_at_dual_drive': (with_burst(box_burst(gain=CARRIER_GAIN_AT_DUAL_DRIVE)), ANATOMICAL, False),
    'gauss250_at_dual_power': (with_burst(gauss_burst(gain=GAUSS250_GAIN_AT_DUAL_POWER)), ANATOMICAL, False),
    'gauss250_at_dual_drive': (with_burst(gauss_burst(gain=GAUSS250_GAIN_AT_DUAL_DRIVE)), ANATOMICAL, False),
    'carrier_only': (regular, ANATOMICAL, False),
    'only_30hz': (with_burst(box_burst(freq=F_LOW, gain=1.0)), ANATOMICAL, False),
    'two_field_250hz': (with_burst(box_burst(gain=TWO_FIELD_GAIN)), (FOCAL_250, FA2), False),
    'constant_substitution': (with_burst(constant_substitution_burst), ANATOMICAL, False),
    'clinical_replica': (clinical_replica, ANATOMICAL, False),
    'clinical_dual_band': (clinical_dual_band, ANATOMICAL, False),
    'both_focal': (dual_band, (FA1, FOCAL_250), False),
    'both_broad': (dual_band, (BROAD_30, FA2), False),
    'dual_band_shifted': (dual_band_phase(90), ANATOMICAL, False),
    'dual_band_phase45': (dual_band_phase(45), ANATOMICAL, False),
    'dual_band_phase135': (dual_band_phase(135), ANATOMICAL, False),
    'dual_band_random_phase': (dual_band_random_phase, ANATOMICAL, False),
    'regular_4': (on_four_sites(box_burst()), ANATOMICAL, False),
    'dual_band_4': (on_four_sites(dual_band_burst()), ANATOMICAL, False),
    'clinical_replica_4': (on_four_sites(box_burst(), clinical=True), ANATOMICAL, False),
    'clinical_dual_band_4': (on_four_sites(dual_band_burst(), clinical=True), ANATOMICAL, False),
}

# number of stimulation sites, where it differs from N_SITES
SITES = {name: FOUR_SITES for name in CONDITIONS if name.endswith('_4')}

MAIN = ['synchronous', 'regular', 'jittered', 'on_off', 'smooth', 'chaotic_timing',
        'chaotic_order', 'interleaved', 'chirped', 'dual_band', 'closed_loop']
DOSE_CONTROLS = ['carrier_at_dual_power', 'carrier_at_dual_drive']
ENVELOPE_CONTROLS = ['gauss250_at_dual_power', 'gauss250_at_dual_drive']
COMPONENT_CONTROLS = ['carrier_only', 'only_30hz', 'two_field_250hz', 'constant_substitution',
                      'clinical_replica', 'dual_band']
FIELD_CONTROLS = ['dual_band', 'both_focal', 'both_broad']
CLINICAL = ['clinical_replica', 'clinical_dual_band']
PHASE = ['dual_band_shifted', 'dual_band_random_phase']
PHASE_SWEEP = ['dual_band_phase45', 'dual_band_phase135']
FOUR_SITE_CLINICAL = ['clinical_replica_4', 'clinical_dual_band_4']
