from pathlib import Path
import sys, os

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import numpy as np

NALL = 40960
dALL = 5120
# N
def expectation_propagation_demod_all(sig_mixture, decod_fun, N):
    if NALL % N != 0:
        print("N should be a multiple of 40960")
        exit(0)

    Nb = NALL // N
    d = N // 8

    est_bits = np.zeros(dALL)
    for n in range(Nb):
        # print(n)
        sig = sig_mixture[n*N: (n+1)*N]
        est_bits[n*d: (n+1)*d] = decod_fun(sig)

    return est_bits

# 3/2 * N
def expectation_propagation_demod_all_overlap(sig_mixture, decod_fun, N):
    if NALL % N != 0:
        print("N should be a multiple of 40960")
        exit(0)

    Nb = NALL // N
    d = N // 8

    est_bits = np.zeros(dALL)

    bits = decod_fun(sig_mixture[0: N+N//2])
    est_bits[0: d] = bits[0: d]

    for n in range(1,Nb-1):
        bits = decod_fun(sig_mixture[n*N-N//4: (n+1)*N+N//4])
        est_bits[n*d: (n+1)*d] = bits[d//4: 5*d//4]

    bits = decod_fun(sig_mixture[(Nb-1)*N-N//2:])
    est_bits[(Nb-1)*d:] = bits[d//2:]

    return est_bits
