import numpy as np
from scipy.signal import coherence, welch

from .params import STIM_DT_S

WINDOW_S = 5.0
HOLD_FRACTION = 0.9

CR_BAND = (0.5, 20.0)
SYNC_BAND = (55.0, 85.0)
N_SECTORS = 5
PAUSE_THRESHOLD_S = 0.040


def window_mean(t, R, cav, t_start, t_end, window=WINDOW_S):
    idx = np.where((t >= max(t_start, t_end - window)) & (t < t_end))[0]
    if len(idx) == 0:
        idx = np.where(t < t_end)[0][-1:]
    return R[idx].mean(), cav[idx].mean()


def outcome_measures(t, R, cav, t_baseline_end, t_stim_end, t_post_end):
    R_base, cav_base = window_mean(t, R, cav, 0.0, t_baseline_end)
    R_on, cav_on = window_mean(t, R, cav, t_baseline_end, t_stim_end)
    R_post, cav_post = window_mean(t, R, cav, t_stim_end, t_post_end)
    return dict(R_baseline=R_base, Cav_baseline=cav_base, R_on=R_on, Cav_on=cav_on,
                R_post=R_post, Cav_post=cav_post, held=bool(R_post < HOLD_FRACTION * R_base))


def band_mean(f, x, band):
    m = (f >= band[0]) & (f <= band[1])
    return float(x[m].mean())


def mean_pair_coherence(signals, fs, nperseg, band=CR_BAND):
    values = []
    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):
            f, C = coherence(signals[i], signals[j], fs=fs, nperseg=nperseg)
            values.append(band_mean(f, C, band))
    return values


def sector_counts(spike_t, spike_i, t0, duration, n, fs=1000.0, n_sectors=N_SECTORS):
    m = (spike_t >= t0) & (spike_t < t0 + duration)
    bins = np.floor((spike_t[m] - t0) * fs).astype(int)
    sector = (spike_i[m] * n_sectors) // n
    counts = np.zeros((n_sectors, int(duration * fs)))
    np.add.at(counts, (sector, bins), 1)
    return counts


def network_coherence(counts, n, fs=1000.0):
    pop = counts.sum(0)
    f, power = welch(pop - pop.mean(), fs=fs, nperseg=4096)
    coh_cr = mean_pair_coherence(counts, fs, 2048, CR_BAND)
    coh_sync = mean_pair_coherence(counts, fs, 2048, SYNC_BAND)
    peak_range = (f >= 40) & (f <= 100)
    summary = dict(sector_coherence=float(np.mean(coh_cr)),
                   sector_coherence_sync_band=float(np.mean(coh_sync)),
                   sync_band_power=band_mean(f, power, SYNC_BAND),
                   sync_peak_hz=float(f[peak_range][np.argmax(power[peak_range])]),
                   mean_rate_hz=counts.sum() / n / (counts.shape[1] / fs))
    return summary, f, power


def long_intervals(spike_t, spike_i, t0, duration, n, threshold=PAUSE_THRESHOLD_S):
    m = (spike_t >= t0) & (spike_t < t0 + duration)
    t, i = spike_t[m], spike_i[m]
    count = 0
    for j in range(n):
        tj = np.sort(t[i == j])
        count += int((np.diff(tj) > threshold).sum())
    return count


def input_coherence(forcing):
    return float(np.mean(mean_pair_coherence(forcing.sum(axis=0), 1 / STIM_DT_S, 2 ** 15)))


def sector_drive(forcing, rf, n_sectors=N_SECTORS):
    drive = np.einsum('cis,cst->it', rf, forcing)
    return drive.reshape(n_sectors, drive.shape[0] // n_sectors, -1).sum(axis=1)


def sector_input_coherence(forcing, rf, n_sectors=N_SECTORS):
    sectors = sector_drive(forcing, rf, n_sectors)
    return float(np.mean(mean_pair_coherence(sectors, 1 / STIM_DT_S, 2 ** 15)))


def own_sector_share(forcing, rf, n_sectors=N_SECTORS):
    """Fraction of each site's input that reaches the sector centred on it, averaged over sites."""
    per_site = np.einsum('cis,cst->is', rf, forcing)
    sectors = per_site.reshape(n_sectors, per_site.shape[0] // n_sectors, -1).sum(axis=1)
    return float(np.mean(np.diag(sectors) / sectors.sum(axis=0)))


def dose(forcing, rf):
    """Drive power and mean drive, both in model units."""
    drive = np.einsum('cis,cst->it', rf, forcing)
    return float(np.mean(np.sum(forcing ** 2, axis=0))), float(drive.mean())
