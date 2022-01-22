from pathlib import Path
import sys, os
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import numpy as np
import scipy

import pickle
import rfcutils

import matplotlib.pyplot as plt


import random
random.seed(0)
np.random.seed(0)

stats_folder = os.path.join('example', 'expectation_propagation', 'stats')
nbSigToEst = 1100
signalLen = 40960
cov_mat_interf = scipy.sparse.eye(signalLen, signalLen)

def load_train_interference(sig_type):
    sig_dataset = []
    for idx in range(nbSigToEst):
        qpsk,meta1,interference,meta2 = rfcutils.load_dataset_sample_components(idx, 'demod_train', sig_type)
        # print(idx, sum(abs(interference)**2)/40960)
        sig_dataset.append(interference)
    sig_dataset = np.array(sig_dataset)
    return sig_dataset

# interference_type = 'EMISignal1'
# interference_sig_dataset = load_train_interference(interference_type)
# mu_interf = np.mean(interference_sig_dataset, axis=0)

# cov_interf_all = np.zeros(signalLen, dtype=np.complex128)
# for i in range(1,nbSigToEst):
#     sig_interf = interference_sig_dataset[i]
#     mean_sig = np.mean(sig_interf)
#     var_sig = np.var(sig_interf)
#     cov_interf = np.correlate(sig_interf - mean_sig, sig_interf - mean_sig, "full")[-signalLen:]
#     cov_interf = cov_interf / (var_sig*np.arange(signalLen,0,-1))
#     print(i, nbSigToEst, cov_interf[0])
#     cov_interf_all += cov_interf
# cov_interf_all = cov_interf_all/nbSigToEst

# cov_interf = cov_interf_all

# pickle.dump((mu_interf,cov_interf),open(os.path.join(stats_folder,f'{interference_type}_stats.pickle'),'wb'))


interference_type = 'CommSignal3'
interference_sig_dataset = load_train_interference(interference_type)
mu_interf = np.mean(interference_sig_dataset, axis=0)

cov_interf_all = np.zeros(signalLen, dtype=np.complex128)
for i in range(1,nbSigToEst):
    sig_interf = interference_sig_dataset[i]
    mean_sig = np.mean(sig_interf)
    var_sig = np.var(sig_interf)
    cov_interf = np.correlate(sig_interf - mean_sig, sig_interf - mean_sig, "full")[-signalLen:]
    # cov_interf = cov_interf / (var_sig*np.arange(signalLen,0,-1))
    cov_interf = cov_interf / var_sig / signalLen
    print(i, nbSigToEst, cov_interf[0])
    cov_interf_all += cov_interf
cov_interf_all = cov_interf_all/nbSigToEst

cov_interf = cov_interf_all

pickle.dump((mu_interf,cov_interf),open(os.path.join(stats_folder,f'{interference_type}_stats.pickle'),'wb'))
