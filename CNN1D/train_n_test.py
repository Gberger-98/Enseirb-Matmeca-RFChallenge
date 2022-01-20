import os, sys
import numpy as np
import pickle

# to run this file from within folder
os.chdir(os.getcwd())
sys.path.append(os.curdir)
import rfcutils
get_sinr = lambda s, i: 10*np.log10(np.mean(np.abs(s)**2)/np.mean(np.abs(i)**2))

import random
random.seed(0)
np.random.seed(0)


from CNN import get_model
from Dataloader import TrainSeq

sig_len = 40960
win_len = 256

#Demodulation
def demod_bits(sig_mixture,model):
    sig_len = len(sig_mixture)
    i = 0
    x_in = []
    tmp = [0]*(win_len//2)
    tmp=np.concatenate((tmp,sig_mixture[0:win_len+win_len//2]))
    x_in.append(tmp)
    i = win_len+win_len//2
    while i+ win_len<sig_len:
        x_in.append(sig_mixture[i-win_len:i+win_len])
        i = i + win_len
    
    tmp = []
    tmp = np.concatenate((tmp,sig_mixture[i-win_len:sig_len]))
    tmp = np.concatenate((tmp,([0]*(win_len*2-len(tmp)))))
    x_in.append(tmp)
    x_in = np.array(x_in)
    x_in = x_in.reshape(-1, 2*win_len)
    x_in_comp = np.stack((x_in.real, x_in.imag), axis=-1)
    
    bit_probs = model.predict(x_in_comp)
    bit_est = np.array(bit_probs > 0.5).flatten()
    return bit_est


#Testing the AI after training
def test(interference_sig_type,nb,model): 
    dataset_type = f'demod_val'
    
    all_ber, all_default_ber, all_sinr = [], [], []
    all_test_idx = np.arange(1100)
    for idx in all_test_idx:
        sig_mixture,meta = rfcutils.load_dataset_sample(idx, dataset_type, interference_sig_type)
        sig1,meta1,sig2,meta2 = rfcutils.load_dataset_sample_components(idx, dataset_type, interference_sig_type)
        
        sinr = get_sinr(sig1, sig2)
        all_sinr.append(sinr)
        ber_ref = rfcutils.demod_check_ber(rfcutils.matched_filter_demod(sig_mixture), idx, dataset_type, interference_sig_type)
        all_default_ber.append(ber_ref)
        
        bit_est1 = demod_bits(sig_mixture,model)
        ber1 = rfcutils.demod_check_ber(bit_est1, idx, dataset_type, interference_sig_type)
        all_ber.append(ber1)
        
        print(f"#{idx} -- SINR {sinr:.3f}dB: 1:{ber1} Default:{ber_ref}")
        
        if len(all_sinr)%100 == 0:
            pickle.dump((all_ber, all_default_ber, all_sinr), open(os.path.join('CNN2','output',f'CNN_{interference_sig_type}_{nb}_val_demod.pickle'),'wb'))
    pickle.dump((all_ber, all_default_ber, all_sinr), open(os.path.join('CNN2','output',f'CNN_{interference_sig_type}_{nb}_val_demod.pickle'),'wb'))


nb_ech = 100
nb_training = 7
nb_epoch = 100
batch_s = 32

sig_type = ['EMISignal1', 'CommSignal2', 'CommSignal3']


isnr = np.arange(-12,4.5,1.5)

for interference_sig_type in sig_type:
    dataloader = TrainSeq(nb_ech,batch_s,win_len,sig_len,interference_sig_type,isnr)
            
    model = get_model(win_len,2)
    for i in range(nb_training):         
        model.fit(dataloader, batch_size=batch_s, epochs=nb_epoch, verbose=1, shuffle=True)
        model.save_weights(os.path.join('CNN2','models',f'CNN_{interference_sig_type}_{i*nb_epoch}'))
        test(interference_sig_type,i*nb_epoch,model)