import os, sys
import numpy as np
import time

import tensorflow as tf

os.chdir(os.getcwd())
sys.path.append(os.curdir)

import rfcutils

import random
import math


class TrainSeq(tf.keras.utils.Sequence):

    def __init__(self, nb_ech, batch_size, window_len,sig_len,interference_sig_type,isnr):
        self.nb_ech = nb_ech
        self.batch_size = batch_size
        self.window_len = 2*window_len
        self.sig_len = sig_len
        self.interference_sig_type = interference_sig_type
        self.n_train_frame_types = {'EMISignal1':530, 'CommSignal2':100, 'CommSignal3':139}
        self.rand = math.floor(time.time())
        self.garde = window_len//16
        self.bit_len = window_len//16*2
        self.isnr = np.random.choice(isnr,nb_ech)

    def __len__(self):
        return self.nb_ech

    def __getitem__(self, idx):
        seed = (idx*self.rand)%(2**31-1)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        x_in, bit_out = [], []
        n_train_frame = self.n_train_frame_types[self.interference_sig_type]
        
        isnr = self.isnr[idx]

        chosen_idx = np.random.randint(n_train_frame)
        data,_ = rfcutils.load_dataset_sample(chosen_idx, 'train_frame', self.interference_sig_type)
        start_idx = np.random.randint(len(data)-self.sig_len)
        sig2 = data[start_idx:start_idx+self.sig_len]
        sig1, _, _, bit_info = rfcutils.generate_qpsk_signal()
        coeff = np.sqrt(np.mean(np.abs(sig1)**2)/(np.mean(np.abs(sig2)**2)*(10**(isnr/10))))
        sig_mixture = sig1 + sig2 * coeff
        for l in range(self.sig_len//self.window_len):
            x_in.append(sig_mixture[l*self.window_len:(l+1)*self.window_len])
            tmp = bit_info[l*self.window_len//16*2:(l+1)*self.window_len//16*2]
            bit_out.append(tmp[self.garde:self.garde+self.bit_len])
        
        x_in = np.array(x_in)
        x_in_comp = np.stack((x_in.real, x_in.imag), axis=-1)
        bit_out = np.array(bit_out)

        return x_in_comp, bit_out
    
    def on_epoch_end(self):
        self.rand = math.floor(time.time())
