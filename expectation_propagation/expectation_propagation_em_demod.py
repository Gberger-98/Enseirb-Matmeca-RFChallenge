from pathlib import Path
import sys, os

import scipy
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import rfcutils

import copy
import numpy as np

from scipy.linalg import det

from scipy.stats import multivariate_normal, norm

import pickle   

import math
import matplotlib.pyplot as plt

from scipy.signal import convolve

from commpy.filters import rrcosfilter, rcosfilter
from commpy.modulation import PSKModem, QAMModem

N = 1024
# N = 768
d = N // 16

INF = 2**64 - 1

# Parameters for QPSK
mod_num = 4
mod = PSKModem(mod_num)

rolloff = 0.5
Fs = 25e6
oversample_factor = 16
Ts = oversample_factor/Fs
tVec, sPSF = rrcosfilter(oversample_factor*d, rolloff, Ts, Fs)
tVec, sPSF = tVec[1:], sPSF[1:]
sPSF = sPSF.astype(np.complex64)

Ov = scipy.sparse.dok_matrix((N,d))
for j in range(0,d):
    Ov[oversample_factor//2 + j*oversample_factor, j] = 1

MT = scipy.linalg.convolution_matrix(sPSF,N,mode='same')
# MT = scipy.linalg.circulant(sPSF)
MT = np.asmatrix(MT)
# y = Hx + s
H = MT @ Ov
H = np.complex128(H)
pinvH = np.linalg.pinv(H)
G = H.H @ H
invG = np.linalg.inv(G)

flatten = lambda x: np.squeeze(np.asarray(x))

def complex_norm(x,mu,v):
    if math.isnan(mu) or math.isnan(v):
        return 0
    p = multivariate_normal.pdf(x.real, mu.real, v) * multivariate_normal.pdf(x.imag, mu.imag, v)
    if math.isnan(p):
        return 0
    return p

def max_etoile2(a,b):
    d = abs(a-b)
    if d < 37:
        d = np.log(1 + np.exp(-d))
    else:
        d = np.log(1 + np.exp(-d))
    return max(a,b) + d

def max_etoile(arr):
    if len(arr) == 2:
        return max_etoile2(arr[0], arr[1])
    else:
        return max_etoile2(arr[0], max_etoile(arr[1:]))


def expectation_propagation_demod(sig_mixture, T=1, negativ_variance_strat=0, interference_sig_type = 'EMISignal1'):
    T = int(T)

    gamma = 1e-6

    Y = np.asmatrix(sig_mixture[0:N]).T
    Y_flat = flatten(Y)

    M = np.asarray([1, -1, 1j, -1j])
    M_size = len(M)

    Id = scipy.sparse.eye(d).todense()

    #####
    # phi, phi_approx, phi_msg
    # psi, psi_approx, phi_msg

    phi = np.ones((d,M_size)) / M_size
    phi_logp = np.ones((d,M_size)) / M_size

    psi = np.ones((d,M_size)) / M_size
    psi_logp = np.ones((d,M_size)) / M_size

    phi_msg_m = np.zeros(d)
    phi_msg_v_s = np.ones(d)

    for t in range(T):
        phi_msg_m_mat = np.asmatrix(phi_msg_m).T
        phi_msg_v = np.diagflat(phi_msg_v_s)

        #### Expectation maximization

        ### OLD METHOD
        # cov_mixture_est = np.correlate(Y_flat, Y_flat, "full")[-N:] / N
        # sigma_w = scipy.linalg.toeplitz(cov_mixture_est) + H @ phi_msg_v @ H.H
        # sigma_w_inv = scipy.linalg.inv(sigma_w)
        # G_tilde = H.H @ sigma_w_inv @ H

        ### NEW METHOD
        for i in range(d):
            for j in range(len(M)):
                s = M[j]
                psi_logp[i,j] = -0.5*abs(s-phi_msg_m[i])**2/phi_msg_v_s[i]

            summ_logp = max_etoile(psi_logp[i,:])
            psi_logp[i,:] -= summ_logp
            psi[i,:] = np.exp(psi_logp[i,:])

        psi_app = np.sum(M * psi, 1)

        interf_est = flatten(Y - H @ np.asmatrix(psi_app).T)
        cov_interf_est = np.correlate(interf_est, interf_est, "full")[-N:] / N
        sigma_w = scipy.linalg.toeplitz(cov_interf_est)
        sigma_w_inv = scipy.linalg.inv(sigma_w)
        G_tilde = H.H @ sigma_w_inv @ H

        #### Rightward message

        # 1) posterior
        sigma_l = scipy.linalg.inv(Id + phi_msg_v @ G_tilde)

        psi_v = sigma_l @ phi_msg_v
        psi_m = sigma_l @ (phi_msg_v @ H.H @ sigma_w_inv @ Y + phi_msg_m_mat)

        # 2) posterior pdf
        psi_approx_m = flatten(psi_m)
        psi_approx_v_s = np.diagonal(psi_v).real

        # 3) extrinsic pdf
        psi_msg_m = (phi_msg_v_s*psi_approx_m - psi_approx_v_s*phi_msg_m) / (phi_msg_v_s - psi_approx_v_s)
        psi_msg_v_s = (phi_msg_v_s*psi_approx_v_s) / (phi_msg_v_s - psi_approx_v_s)


        #### Leftward message

        # 1) posterior pmf
        for i in range(d):
            for j in range(len(M)):
                s = M[j]
                phi_logp[i,j] = -0.5*abs(s-psi_msg_m[i])**2/psi_msg_v_s[i]

            summ_logp = max_etoile(phi_logp[i,:])
            phi_logp[i,:] -= summ_logp
            phi[i,:] = np.exp(phi_logp[i,:])

        # 2) posterior pdf
        phi_approx_m = np.sum(M * phi, 1)
        phi_approx_v_s = np.sum(abs(M)**2 * phi, 1) - abs(phi_approx_m)**2
        
        if t+1 == T:
            break

        # 3) extrinsic pdf
        phi_msg_m = (psi_msg_v_s*phi_approx_m - phi_approx_v_s*psi_msg_m) / (psi_msg_v_s - phi_approx_v_s)
        phi_msg_v_s = (psi_msg_v_s*phi_approx_v_s) / (psi_msg_v_s - phi_approx_v_s)

        for i in range(d):
            if psi_msg_v_s[i] - phi_approx_v_s[i] < gamma:
                phi_msg_m[i] = phi_approx_m[i]
                phi_msg_v_s[i] = phi_approx_v_s[i]


    # compute bits
    est_symb = phi.argmax(1)
    est_bits = mod.demodulate(M[est_symb], 'hard')
    # est_bits = mod.demodulate(phi_approx_m, 'hard')
    return est_bits

