'''
trying to build a GAN instead of manually generating attacks

train model that generates attacks
implant attacks in clean data, write to disc
'''

from __future__ import absolute_import, division, print_function, unicode_literals

from scipy.stats import multivariate_normal

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from keras.layers import Input, Lambda, Dense, BatchNormalization, Activation, Dropout, Conv1D, Flatten, MaxPooling1D, UpSampling1D
from keras.models import Model, Sequential
from keras.losses import mse, categorical_crossentropy, binary_crossentropy
from keras.utils import plot_model, to_categorical
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import sgd
from keras.callbacks import EarlyStopping

import tensorflow as tf

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

import attack-generator-model

'''
CONSTANTS
'''

BUFFER_SIZE = 60000
BATCH_SIZE = 256

'''
TREAT DATA
'''

def roll(x, y, window_length):
    """
    roll data into windows
    """
    n = x.shape[0] - window_length + 1 # new num obs
    m = x.shape[1] # num features
    x_rolled = np.zeros((n,window_length,m))
    y_rolled = np.zeros(n)
    for w in range(n):
        x_rolled[w] = x[w:(w+window_length)]
        y_rolled[w] = np.any(y[w:(w+window_length)]==1).astype(int)

    return (x_rolled, y_rolled)

'''
DEFINE SIMULTANEOUS TRAINING REGIME
'''

@tf.function
# apparently this needs to compile before we train

def train_step(generator, discriminator,
               attack_data, clean_data):
    """
    one training step
    updates gradients of generator and discriminator at same time
    """

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      synthetic_attack_data=generator(clean_data, training=True)

      real_output = discriminator(attack_data, training=True)
      fake_output = discriminator(synthetic_attack_data, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(attack_data, epochs):
    """
    train generator and discriminator in batches
    """

    for epoch in range(epochs):
        start=time.time()
        for batch in attack_data:
            train_step(generator, discriminator,
                batch, clean_data.sample(n=BATCH_SIZE))

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))



'''
IMPORT AND PREPROCESS DATA
'''

W_LENGTH = 24       # time window of each rolled observation
STD_PCT = 0.1       # percentage of features to toss, determined by lowest variance

# import
clean = pd.read_csv("./batadal/BATADAL_dataset03.csv")
attack = pd.read_csv("./batadal/BATADAL_dataset04_manualflags.csv")

# feature/response format
y_clean = clean['ATT_FLAG'].values
y_attack = attack['ATT_FLAG'].values
clean = clean.drop(columns=['DATETIME','ATT_FLAG'])
attack = attack.drop(columns=['DATETIME','ATT_FLAG']).values

names = clean.columns
clean = clean.values            # keras takes np arrays

# standardize data and remove low-variance features
scaler = MinMaxScaler()
scaler.fit(clean)
clean = scaler.fit_transform(clean)
attack = scaler.transform(attack)
keep = clean.std(0) > np.percentile(clean.std(0), STD_PCT)
clean = clean[:,keep]
attack = attack[:,keep]
names = names[keep]

# roll data into windows
clean_roll, y_clean_roll = roll(clean, y_clean, W_LENGTH)
attack_roll, y_attack_roll = roll(attack, y_attack, W_LENGTH)

'''
MAIN LOOP
BUILD THE SYNTHETIC DATASET
'''

def main():
    """
    Train, implant, and write to disc.
    """
    # instantiate models from attack-generator-model.py
    generator = 0
    discriminator = 0
    # train up to a certain point on clean data
    # batch and shuffle attack data
    # train the models to generate synthetic attacks
    # generate attacks using the decoder & random noise as input
    # implant synthetic attacks
    # write new dataset to disc


if __name__ == '__main__':
    main()
