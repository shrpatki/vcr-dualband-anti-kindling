import numpy as np

N = 80
N_SITES = 5

DT = 0.01                                 # ms
STIM_DT_MS = 0.1
STIM_STRIDE = int(round(STIM_DT_MS / DT))
STIM_DT_S = STIM_DT_MS / 1000.0
SAVE_STRIDE = 50                          # R and Cav are stored every 0.5 ms

C_MEM = 1.0                               # uF/cm2
G_NA, G_K, G_L = 120.0, 36.0, 0.3         # mS/cm2
V_NA, V_K, V_L = 50.0, -77.0, -54.4       # mV
I0, DELTA_I = 11.0, 0.45                  # uA/cm2, drawn uniformly from I0 +- DELTA_I

RING_LENGTH = 10.0
A1, A2 = 3.5, 2.0
V_SYN_EXC, V_SYN_INH = 20.0, -40.0        # mV

BETA1, BETA2 = 1.0, 16.0
GAMMA1, GAMMA2 = 0.12, 0.15
TAU_STDP = 14.0                           # ms
ETA = 0.002
C_MAX = 1.0

V_STIM = 20.0                             # mV
K_STIM = 0.35                             # mS/cm2

PROTOCOL = (2.0, 60.0, 128.0, 128.0)      # s: equilibration, baseline, stimulation, post

EMA_TAU_MS = 500.0
EMA_THRESHOLD = 0.5


def neuron_positions(n=N):
    return (np.arange(n) + 0.5) / n


def site_positions(n_sites=N_SITES):
    return (np.arange(n_sites) + 0.5) / n_sites
