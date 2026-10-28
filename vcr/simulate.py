import numpy as np
from numba import njit

from .neuron import step_neuron, step_gate
from .network import apply_stdp


@njit(fastmath=True)
def stimulus_drive(forcing, rf, k):
    n_channels, n, n_sites = rf.shape
    drive = np.zeros(n)
    for i in range(n):
        acc = 0.0
        for c in range(n_channels):
            for site in range(n_sites):
                acc += rf[c, i, site] * forcing[c, site, k]
        drive[i] = acc
    return drive


@njit(fastmath=True)
def synaptic_current(V, c, M, V_syn, q):
    n = V.shape[0]
    S = np.zeros(n)
    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc += (V_syn[i, j] - V[i]) * c[i, j] * abs(M[i, j]) * q[j]
        S[i] = acc / n
    return S


@njit(fastmath=True)
def order_parameter(t_now, last_spike, prev_spike):
    n = last_spike.shape[0]
    sum_cos, sum_sin, valid = 0.0, 0.0, 0
    for i in range(n):
        if last_spike[i] > -1.0e8 and prev_spike[i] > -1.0e8:
            isi = last_spike[i] - prev_spike[i]
            if isi > 1.0e-6:
                phase = 2 * np.pi * (t_now - last_spike[i]) / isi
                sum_cos += np.cos(phase)
                sum_sin += np.sin(phase)
                valid += 1
    if valid == 0:
        return 0.0
    return np.sqrt(sum_cos ** 2 + sum_sin ** 2) / valid


@njit(fastmath=True)
def mean_signed_weight(c, M):
    n = c.shape[0]
    acc = 0.0
    for i in range(n):
        for j in range(n):
            if M[i, j] > 0:
                acc += c[i, j]
            elif M[i, j] < 0:
                acc -= c[i, j]
    return acc / (n * n)


@njit(fastmath=True)
def weighted_signed_weight(c, M):
    """Mean signed weight with each pair weighted by |M_ij|."""
    n = c.shape[0]
    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(n):
            a = abs(M[i, j])
            if a == 0.0:
                continue
            s = 1.0 if M[i, j] > 0 else -1.0
            num += s * a * c[i, j]
            den += a
    if den == 0.0:
        return 0.0
    return num / den


@njit(fastmath=True)
def run_protocol(I_drive, M, V_syn, c0, forcing, rf,
                 t_equil, t_baseline, t_stim, t_post, dt,
                 k_stim, v_stim, beta1, beta2, gamma1, gamma2, tau, eta, c_max,
                 save_stride, seed, stim_stride,
                 closed_loop=False, ema_tau=500.0, ema_threshold=0.5,
                 record_spikes=False, spike_capacity=200000, spike_window=2000.0):
    """Integrate the network through the full protocol (durations in ms).

    Returns R(t), Cav(t), the step indices where stimulation starts and ends,
    the total number of steps, the weight matrices at the end of baseline,
    stimulation and post-stimulation, and, if requested, spike times and
    indices in windows of `spike_window` ms before stimulation onset, after
    onset and after offset.
    """
    np.random.seed(seed)
    n = I_drive.shape[0]
    n_forcing = forcing.shape[2]

    V = np.empty(n)
    for i in range(n):
        V[i] = -65.0 + 70.0 * np.random.random()
    m = np.random.random(n)
    h = np.random.random(n)
    g = np.random.random(n)
    q = np.random.random(n)
    c = c0.copy()

    n_equil = int(round(t_equil / dt))
    n_baseline = int(round(t_baseline / dt))
    n_stim = int(round(t_stim / dt))
    n_post = int(round(t_post / dt))
    n_steps = n_equil + n_baseline + n_stim + n_post
    on_start = n_equil + n_baseline
    on_end = on_start + n_stim

    last_spike = np.full(n, -1.0e9)
    prev_spike = np.full(n, -1.0e9)

    n_saved = n_steps // save_stride + 1
    R_hist = np.zeros(n_saved)
    cav_hist = np.zeros(n_saved)
    saved = 0
    drive = np.zeros(n)
    r_ema = 1.0
    ema_alpha = dt / ema_tau

    c_baseline_end = c0.copy()
    c_stim_end = c0.copy()

    win = int(round(spike_window / dt))
    pre_start = max(on_start - win, n_equil)
    on_win_end = min(on_start + win, on_end)
    off_win_end = min(on_end + win, n_steps)
    spike_t = np.full(spike_capacity, -1.0)
    spike_i = np.full(spike_capacity, -1)
    n_recorded = 0

    for step in range(n_steps):
        t_now = step * dt
        plastic = step >= n_equil
        stimulating = on_start <= step < on_end

        if step == on_start:
            c_baseline_end = c.copy()
        if step == on_end:
            c_stim_end = c.copy()

        if stimulating:
            k = min((step - on_start) // stim_stride, n_forcing - 1)
            drive = stimulus_drive(forcing, rf, k)

        gate_open = True
        if stimulating and closed_loop:
            r_now = order_parameter(t_now, last_spike, prev_spike)
            r_ema = r_ema + ema_alpha * (r_now - r_ema)
            gate_open = r_ema >= ema_threshold

        S = synaptic_current(V, c, M, V_syn, q)

        in_window = record_spikes and (
            (pre_start <= step < on_start)
            or (on_start <= step < on_win_end)
            or (on_end <= step < off_win_end)
        )

        for i in range(n):
            I_stim = (v_stim - V[i]) * k_stim * drive[i] if (stimulating and gate_open) else 0.0
            V_new, m[i], h[i], g[i] = step_neuron(V[i], m[i], h[i], g[i], I_drive[i], S[i], I_stim, dt)
            q[i] = step_gate(V[i], q[i], dt)

            spiked = (V_new > 0.0) and (V[i] <= 0.0)
            V[i] = V_new
            if spiked:
                if plastic:
                    apply_stdp(c, M, i, t_now, last_spike, beta1, beta2, gamma1, gamma2, tau, eta, c_max)
                prev_spike[i] = last_spike[i]
                last_spike[i] = t_now
                if in_window and n_recorded < spike_capacity:
                    spike_t[n_recorded] = t_now
                    spike_i[n_recorded] = i
                    n_recorded += 1

        if step % save_stride == 0:
            R_hist[saved] = order_parameter(t_now, last_spike, prev_spike)
            cav_hist[saved] = mean_signed_weight(c, M)
            saved += 1

    return (R_hist[:saved], cav_hist[:saved], on_start, on_end, n_steps,
            c_baseline_end, c_stim_end, c.copy(),
            spike_t[:n_recorded], spike_i[:n_recorded])
