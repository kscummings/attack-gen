'''
trying to build a GAN instead of manually generating attacks

train model that generates attacks
implant attacks in clean data, write to disc
'''

from __future__ import absolute_import, division, print_function, unicode_literals

from scipy.stats import multivariate_normal

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Lambda, Dense, BatchNormalization, Activation, Dropout, Conv1D, Flatten, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse, categorical_crossentropy, binary_crossentropy
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
#from keras.optimizers import sgd
#from keras.callbacks import EarlyStopping

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time

import attack-GAN
import format-data

'''
CONSTANTS
'''

BUFFER_SIZE = 60000
BATCH_SIZE = 256
TEST_SIZE = 0.3
EPOCHS = 20

'''
DEFINE SIMULTANEOUS TRAINING REGIME
'''

@tf.function # apparently this needs to compile before we train
def train_step(attack_data):
    """
    one training step
    updates gradients of generator and discriminator at same time
    """
    noise = tf.random.normal([BATCH_SIZE, 6, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      synthetic_attack_data=generator(noise, training=True)

      real_output = discriminator(attack_data, training=True)
      fake_output = discriminator(synthetic_attack_data, training=True)

      gen_loss = attack_GAN.generator_loss(fake_output)
      disc_loss = attack-GAN.discriminator_loss(real_output, fake_output)

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
            train_step(generator, discriminator, batch)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))



'''
MAIN LOOP
BUILD THE SYNTHETIC DATASET
'''

if __name__ == '__main__':

    """
    Train, implant, and write to disc.
    """

    # get data
    (clean, y_clean), (attack, y_attack), names = format-data.get_rolled_data()

    # prime the decoder by training the VAE to reconstruct clean obs
    clean_train, clean_val = train_test_split(clean, test_size=TEST_SIZE, shuffle=True)
    _, decoder, vae = attack-GAN.vae_gen_model((clean_train, clean_val))

    generator = decoder
    discriminator = attack-GAN.disc_model()
    generator_optimizer=tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)

    # batch and shuffle attack data
    train_attacks = format-data.get_rolled_attack_data()
    train_dataset = tf.data.Dataset.from_tensor_slices(train_attacks).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # train the models to generate synthetic attacks
    train(train_dataset, EPOCHS)

    # implant synthetic attacks
    # write new dataset to disc
