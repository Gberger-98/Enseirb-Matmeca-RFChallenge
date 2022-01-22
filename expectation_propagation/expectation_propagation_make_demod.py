import os, sys
from matplotlib.pyplot import switch_backend
os.chdir(os.getcwd())
print(os.path.abspath(os.curdir))
sys.path.append(os.curdir)

import matplotlib.pyplot as plt

from random import randint

import numpy as np
import pickle

import rfcutils
get_pow = lambda s: np.mean(np.abs(s)**2)
get_sinr = lambda s1, s2: 10*np.log10(get_pow(s1)/get_pow(s2))

import random
random.seed(0)
np.random.seed(0)

val_or_test = 'val'

output_folder = os.path.join('expectation_propagation', 'output')



################## PARAMS ##################
T = 4               # number of iterations
TYPE = 1            # 0: ep, 1:ep+em

ALLSIG = False      # use entire signal
OVERLAP = False     # use overlap

N = 1024            # subset of the signal considered by the algo
"""
if ALLSIG is set to False, then N in the used algorithm should set to the same value: N aswell
if ALLSIG AND OVERLAP are set to True, then N in the used algorithm should set to 3*N/2
"""

interference_sig_type = 'EMISignal1'
# interference_sig_type = 'CommSignal2'
# interference_sig_type = 'CommSignal3'
############################################


# do not touch
NAMEALLSIG = ""
if ALLSIG == True:
    from ep_demod_all import *
    if OVERLAP == False:
        NAMEALLSIG = f'ALLSIG_{N}_'
    else:
        NAMEALLSIG = f'ALLSIG_OVERLAP_{N}_'

if TYPE == 0: #EP using database
    print('EP USING DATABASE')
    from expectation_propagation_demod import expectation_propagation_demod
    NAME = "EP"
elif TYPE == 1: #EP with Expectation Maximization (EM)
    print('EP WITH EM')
    from expectation_propagation_em_demod import expectation_propagation_demod
    NAME = "EP_EM"
else:
    print('invalid type')
    exit(0)


def main():
    dataset_type = f'demod_{val_or_test}'
    all_ber, all_default_ber, all_sdr, all_sinr = [], [], [], []
    
    all_test_idx = np.arange(0, 1100)
    for idx in all_test_idx:
        sig_mixture,meta = rfcutils.load_dataset_sample(idx, dataset_type, interference_sig_type)
        sig1,meta1,sig2,meta2 = rfcutils.load_dataset_sample_components(idx, dataset_type, interference_sig_type)
        
        sinr = get_sinr(sig1, sig2)
        all_sinr.append(sinr)
        ber_ref = rfcutils.demod_check_ber(rfcutils.matched_filter_demod(sig_mixture), idx, dataset_type, interference_sig_type)
        all_default_ber.append(ber_ref)

        
        if ALLSIG == False:
            bit_est = expectation_propagation_demod(sig_mixture, 
                T=T,
                interference_sig_type = interference_sig_type
            )
        elif ALLSIG == True:
            decod_fun = lambda sig: expectation_propagation_demod(sig, 
                T=T,
                interference_sig_type = interference_sig_type
            )
            if not OVERLAP:
                bit_est = expectation_propagation_demod_all(sig_mixture, decod_fun, N)
            else:
                bit_est = expectation_propagation_demod_all_overlap(sig_mixture, decod_fun, N)

        

        ber = rfcutils.demod_check_ber(bit_est, idx, dataset_type, interference_sig_type)
        all_ber.append(ber)
        
        sdr = 0
        all_sdr.append(sdr)
        
        print(f"#{idx} -- SINR {sinr:.3f}dB: 1:{ber} Default:{ber_ref}, SDR:{sdr}")
        
        if len(all_sinr)%100 == 0:
            pickle.dump((all_ber, all_default_ber, all_sdr, all_sinr), open(os.path.join(output_folder, f'expectation_propagation_T_{T}_{NAMEALLSIG}{NAME}_{interference_sig_type}_{val_or_test}_demod.pickle'),'wb'))
    pickle.dump((all_ber, all_default_ber, all_sdr, all_sinr), open(os.path.join(output_folder, f'expectation_propagation_T_{T}_{NAMEALLSIG}{NAME}_{interference_sig_type}_{val_or_test}_demod.pickle'),'wb'))

    
if __name__ == "__main__":
    main()