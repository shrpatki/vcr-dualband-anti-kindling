import numpy as np
from numba import njit

from .params import C_MEM, G_NA, G_K, G_L, V_NA, V_K, V_L


@njit(fastmath=True)
def alpha_m(V):
    if abs(V + 40.0) < 1e-7:
        return 1.0
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))


@njit(fastmath=True)
def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


@njit(fastmath=True)
def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


@njit(fastmath=True)
def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


@njit(fastmath=True)
def alpha_n(V):
    if abs(V + 55.0) < 1e-7:
        return 0.1
    return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))


@njit(fastmath=True)
def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


@njit(fastmath=True)
def step_neuron(V, m, h, n, I_drive, S, I_stim, dt):
    am, bm = alpha_m(V), beta_m(V)
    ah, bh = alpha_h(V), beta_h(V)
    an, bn = alpha_n(V), beta_n(V)
    dV = (I_drive - G_NA * m ** 3 * h * (V - V_NA) - G_K * n ** 4 * (V - V_K)
          - G_L * (V - V_L) + S + I_stim) / C_MEM
    V_new = V + dt * dV
    m_new = min(max(m + dt * (am * (1 - m) - bm * m), 0.0), 1.0)
    h_new = min(max(h + dt * (ah * (1 - h) - bh * h), 0.0), 1.0)
    n_new = min(max(n + dt * (an * (1 - n) - bn * n), 0.0), 1.0)
    return V_new, m_new, h_new, n_new


@njit(fastmath=True)
def step_gate(V, q, dt):
    dq = 0.5 * (1 - q) / (1.0 + np.exp(-(V + 5.0) / 12.0)) - 2.0 * q
    return min(max(q + dt * dq, 0.0), 1.0)
