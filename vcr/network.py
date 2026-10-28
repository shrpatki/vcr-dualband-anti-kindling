import numpy as np
from numba import njit

from .params import RING_LENGTH, A1, A2, V_SYN_EXC, V_SYN_INH, C_MAX


def mexican_hat(n, a1=A1, a2=A2, ring_length=RING_LENGTH):
    idx = np.arange(n)
    diff = np.abs(idx[:, None] - idx[None, :])
    diff = np.minimum(diff, n - diff)
    d = ring_length / (n - 1) * diff
    M = (1 - d ** 2 / a1 ** 2) * np.exp(-d ** 2 / (2 * a2 ** 2))
    np.fill_diagonal(M, 0.0)
    return M


def synaptic_reversal(M):
    return np.where(M > 0, V_SYN_EXC, V_SYN_INH)


def coupling_bounds(M, c_max=C_MAX):
    """Smallest and largest attainable mean signed coupling for a connectivity matrix."""
    n = M.shape[0]
    return -(M < 0).sum() * c_max / n ** 2, (M > 0).sum() * c_max / n ** 2


@njit(fastmath=True)
def stdp_kernel(dt, beta1, beta2, gamma1, gamma2, tau):
    if dt >= 0.0:
        return beta1 * np.exp(-dt / (gamma1 * tau))
    return beta2 * (dt / tau) * np.exp(dt / (gamma2 * tau))


@njit(fastmath=True)
def apply_stdp(c, M, i, t_now, last_spike, beta1, beta2, gamma1, gamma2, tau, eta, c_max):
    n = c.shape[0]
    for j in range(n):
        if j == i or last_spike[j] <= -1.0e8:
            continue
        s_ij = 1.0 if M[i, j] > 0 else -1.0
        w = stdp_kernel(t_now - last_spike[j], beta1, beta2, gamma1, gamma2, tau)
        c[i, j] = min(max(c[i, j] + s_ij * eta * w, 0.0), c_max)

        s_ji = 1.0 if M[j, i] > 0 else -1.0
        w = stdp_kernel(last_spike[j] - t_now, beta1, beta2, gamma1, gamma2, tau)
        c[j, i] = min(max(c[j, i] + s_ji * eta * w, 0.0), c_max)
