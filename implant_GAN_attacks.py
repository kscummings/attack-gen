'''
build a GAN instead of manually generating attacks
implant synthetic attacks in clean data, write to disc
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
import argparse

import attack_GAN
import format_data

'''
CONSTANTS
'''

BUFFER_SIZE = 60000
BATCH_SIZE = 256
TEST_SIZE_CL = 0.3
TEST_SIZE_ATT = 0.15
EPOCHS_VAE = 200
EPOCHS_GAN = 10

'''
DEFINE SIMULTANEOUS TRAINING REGIME
'''

@tf.function # apparently this needs to compile before we train
def train_step(attack_data):
    """
    one training step
    updates gradients of generator and discriminator at same time
    """
    noise = tf.random.normal([attack_data.shape[0], 6, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        _,_,synthetic_attack_data=generator(noise, training=True)

        real_output = discriminator(attack_data, training=True) # note: doesn't like np array inputs
        fake_output = discriminator(synthetic_attack_data, training=True)

        gen_loss = attack_GAN.generator_loss(fake_output)
        disc_loss = attack_GAN.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(attack_data, val_attack_data, epochs):
    """
    train generator and discriminator in batches
    """

    for epoch in range(epochs):
        start=time.time()
        for batch in attack_data:
            g,d=train_step(batch)

        g,d=np.array(g),np.array(d) # per-instance training loss
        gen_loss,disc_loss=sum(g),sum(d)

        # get val loss
        noise=tf.random.normal([val_attack_data.shape[0], 6, 100])
        _,_,val_synthetic_data=generator(noise,training=False)
        real_output=discriminator(val_attack_data,training=False)
        fake_output=discriminator(val_synthetic_data,training=False)

        val_gen_loss = sum(np.array(attack_GAN.generator_loss(fake_output)))
        val_disc_loss = sum(np.array(attack_GAN.discriminator_loss(real_output,fake_output)))

        history=np.stack([gen_loss,disc_loss,val_gen_loss,val_disc_loss]).T
        loss_history=(history if epoch==0 else np.concatenate((loss_history,history),axis=0))
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print('Gen/disc loss: {}, {}. Val gen/disc loss: {}, {}.'.format(gen_loss, disc_loss,
            val_gen_loss, val_disc_loss))



'''
MAIN LOOP
BUILD THE SYNTHETIC DATASET
'''

if __name__ == '__main__':

    """
    Train, implant, and write to disc.
    """
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights for decoder"
    parser.add_argument("-w", "--weights", help=help_)
    args = parser.parse_args()

    # get data
    print("\n Getting data \n \n")
    (clean, y_clean), (attack, y_attack), names = format_data.get_rolled_data()

    # prime the decoder by training the VAE to reconstruct clean obs
    clean_train, clean_val = train_test_split(clean, test_size=TEST_SIZE_CL, shuffle=True)
    if args.weights:
        print("\n Loading the primed generator \n \n ")
        _, decoder, _ = attack_GAN.vae_gen_model((clean_train, clean_val),to_train=False)
        decoder.load_weights(args.weights)
    else:
        print("\n Training the VAE (priming generator) \n \n")
        _, decoder, _ = attack_GAN.vae_gen_model((clean_train, clean_val),ep=EPOCHS_VAE)

    # set up the GAN
    generator = decoder
    discriminator = attack_GAN.disc_model()
    generator_optimizer=tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)

    # train the GAN
    print("\n Training the GAN \n \n")
    attacks = format_data.get_rolled_attack_data()
    train_attacks, val_attacks = train_test_split(attacks, test_size=TEST_SIZE_ATT, shuffle=True)
    train_attacks, val_attacks = tf.cast(train_attacks,tf.float32), tf.cast(val_attacks,tf.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_attacks).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train(train_dataset, val_attacks, EPOCHS_GAN)

    # implant synthetic attacks
    # write new dataset to disc
