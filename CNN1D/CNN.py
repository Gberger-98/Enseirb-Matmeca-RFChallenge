import os, sys
import tensorflow as tf
from tensorflow.keras import layers,models
import numpy as np

os.chdir(os.getcwd())
sys.path.append(os.curdir)


def get_model(n = 512, n_ch=2):
    model = models.Sequential(layers.Dense(n, activation="relu",input_shape=(n*2, n_ch)))
    # model.add(layers.Flatten())
    for i in range(5):
        model.add(layers.Conv1D(2**(8-i), 16, activation='relu'))
        model.add(layers.Conv1D(2**(8-i), 16, activation='relu'))
        model.add(layers.Conv1D(2**(7-i), 16, activation='relu'))
        model.add(layers.MaxPool1D(pool_size=2,strides=1, padding='valid'))
    model.add(layers.Flatten())
    model.add(layers.Dense(n//16*2,activation='sigmoid'))



    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005),
        metrics=["binary_accuracy"],
    )

    return model

