'''
trying to build a GAN instead of manually generating attacks

train model that generates attacks
implant attacks in clean data, write to disc
'''

from __future__ import absolute_import, division, print_function, unicode_literals

from scipy.stats import multivariate_normal

from sklearn.metrics import confusion_matrix
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

import attack-GAN
import format-data

'''
CONSTANTS
'''

BUFFER_SIZE = 60000
BATCH_SIZE = 256


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
MAIN LOOP
BUILD THE SYNTHETIC DATASET
'''

def main():
    """
    Train, implant, and write to disc.
    """

    # get data
    (clean_roll, y_clean_roll), (attack_roll, y_attack_roll) = format-data.get_rolled_data()

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
