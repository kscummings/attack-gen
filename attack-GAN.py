'''
generator and discriminator to generate synthetic attacks
training regime code based on https://www.tensorflow.org/alpha/tutorials/generative/dcgan
VAE generator based on https://ascelibrary.org/doi/full/10.1061/%28ASCE%29WR.1943-5452.0001007
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.stats import multivariate_normal

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Lambda, Dense, BatchNormalization, Activation, Dropout, Conv1D, Flatten, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse, categorical_crossentropy, binary_crossentropy
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
#from tensorflow.keras.optimizers import sgd
#from tensorflow.keras.callbacks import EarlyStopping

#import seaborn as sns
import numpy as np
import pandas as pd
# uncomment if in virtualenv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import os

'''
BUILD AND PRIME GENERATOR
'''

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent array of size .... um
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim1 = K.int_shape(z_mean)[1]
    dim2 = K.int_shape(z_mean)[2]
    # by default, random_normal has mean=0 and sd=1.0
    epsilon = K.random_normal(shape=(batch, dim1, dim2))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon # latent tensor


def vae_gen_model(data,                  # unroll into train/validation feat.
                  input_shape=(24,36),
                  k_size=5,
                  conv=[60,80,36],
                  resolution=2,
                  dense=[200,160,480],
                  latent_dim=100,
                  pad_type='same',
                  act_type='relu',
                  ep=40,
                  b_s=128):
    """
    Prime attack generator to reconstruct clean windows
    Architecture is Chandy's VAE
    Train in function to ensure decoder is trained and separable
    """
    (clean_train, clean_val) = data

    # build encoder

    conv[2] = input_shape[1]
    inputs = Input(shape=input_shape,name='encoder_input')
    conv_1 = Conv1D(filters = conv[0], kernel_size = k_size,
             padding = pad_type, strides = 1)(inputs)
    batch_1 = BatchNormalization(axis = 2)(conv_1)    # channels along last axis
    pool_1 = MaxPooling1D(pool_size = resolution,padding = pad_type)(batch_1)
    conv_2 = Conv1D(filters = conv[1], kernel_size = k_size, padding = pad_type,
             strides = 1)(pool_1)
    batch_2 = BatchNormalization(axis = 2)(conv_2)
    pool_2 = MaxPooling1D(pool_size = resolution,padding = pad_type)(batch_2)
    dense_1 = Dense(units = dense[0], activation = act_type)(pool_2)
    batch_3 = BatchNormalization(axis = 2)(dense_1)
    dense_2 = Dense(units = dense[1], activation = act_type)(batch_3)
    batch_4 = BatchNormalization(axis = 2)(dense_2)
    z_mean_un = Dense(units = latent_dim)(batch_4)
    z_log_var_un = Dense(units = latent_dim, activation = act_type)(batch_4)
    z_mean = BatchNormalization(axis = 2)(z_mean_un)
    z_log_var = BatchNormalization(axis = 2)(z_log_var_un)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    # output shape is (batch_size, 6, latent_dim)
    z = Lambda(function = sampling, name='z')([z_mean, z_log_var])

    # build decoder
    latent_inputs = Input(shape=K.int_shape(z)[1:], name='z_sampling')
    d_dense_1 = Dense(units=dense[1], activation=act_type)(latent_inputs)
    d_batch_1 = BatchNormalization(axis=2)(d_dense_1)
    d_dense_2 = Dense(units=dense[0], activation=act_type)(d_batch_1)
    d_batch_2 = BatchNormalization(axis=2)(d_dense_2)
    d_dense_3 = Dense(units=dense[2], activation=act_type)(d_batch_2)
    d_batch_3 = BatchNormalization(axis=2)(d_dense_3)
    d_up_1 = UpSampling1D(size=resolution)(d_batch_3)
    d_conv_1 = Conv1D(filters=conv[0], kernel_size = k_size,
               padding = pad_type, strides = 1)(d_up_1) #data_format = "channels_last"
    d_batch_4 = BatchNormalization(axis=2)(d_conv_1)
    d_up_2 = UpSampling1D(size=resolution)(d_batch_4)
    d_mean_un = Conv1D(filters=conv[2], kernel_size = k_size,
                padding = pad_type, strides = 1)(d_up_2) # data_format = "channels_last"
    d_log_var_un = Conv1D(filters=conv[2], kernel_size = k_size,
                   padding = pad_type, strides = 1)(d_up_2) #data_format = "channels_last"
    d_mean = BatchNormalization(axis = 2)(d_mean_un)
    d_log_var = BatchNormalization(axis=2)(d_log_var_un)

    # join the branches by generating random normals. this is our reconstruction
    outputs = Lambda(function = sampling, name='output')([d_mean, d_log_var])

    # instantiate models
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(latent_inputs, [d_mean,d_log_var,outputs], name='decoder')
    outputs = decoder(encoder(inputs)[2])[2]
    vae = Model(inputs, outputs, name='vae')

    # add loss and optimizer to vae
    #reconstruction_loss = mse(inputs, outputs)
    # reconstruction_loss = mse(inputs, outputs)
    # reconstruction_loss *= input_shape[1]
    # reconstruction_loss = K.sum(reconstruction_loss, axis=-1)
    #
    # kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    # kl_loss = K.sum(kl_loss, axis=-1)
    # kl_loss = K.sum(kl_loss, axis=-1)
    # kl_loss *= -0.5
    #
    # vae_loss = K.mean(reconstruction_loss + kl_loss)

    v_loss = vae_loss(inputs, outputs, input_shape, z_mean, z_log_var)
    vae.add_loss(v_loss)
    vae.compile(optimizer='adam')

    vae.fit(clean_train,epochs=ep,batch_size=b_s,validation_data=(clean_val,None))

    return encoder, decoder, vae



'''
BUILD DISCRIMINATOR
'''

def disc_model(conv_layers = [50,50,50],
                 dense_layers = [10,10],
                 k_size = 5,
                 pad_type = 'same',
                 resolution = 2,
                 kernel_initializer = 'glorot_uniform',
                 num_classes = 1,
                 input_shape = (24,36),
                 learning_rate = .01,
                 reg = .00):
    """
    Build a discriminator that tries to differentiate between
    real and fake attacks
    Based closely on the anomaly detection architecture
    # Arguments
        conv_layers = number of filters in each conv layer (at least one layer)
        dense_layers = dimension of each dense layer (at least one layer)
        k_size = convolutional filter size
        pad_type = zero-padding to use during convolution
        resolution = downsampling factor in max pool layers
        kernel_initializer = NN parameter initialization or something like that idk
        num_classes = output dim of each obs
        input_shape = input dim of each obs
        learning_rate = self-explanatory
        reg = self-explanatory
    """
    disc = Sequential()
    disc.add(Conv1D(input_shape=input_shape,
                  filters = conv_layers[0],
                  kernel_size = k_size,
                  padding = pad_type,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer = l2(reg),
                  bias_initializer = 'ones',
                  bias_regularizer = l2(reg)))
    disc.add(MaxPooling1D(pool_size = resolution, padding = pad_type))
    disc.add(BatchNormalization(axis=2))

    n_conv = len(conv_layers)
    n_dense = len(dense_layers)

    for i in range(1, n_conv):
        disc.add(Conv1D(filters = conv_layers[i],
                  kernel_size = k_size,
                  padding = pad_type,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer = l2(reg),
                  bias_initializer = 'ones',
                  bias_regularizer = l2(reg)
                       )
                )
        disc.add(MaxPooling1D(pool_size = resolution, padding = pad_type))
        disc.add(BatchNormalization(axis=2))

    disc.add(Flatten())

    for i in range(n_dense):
        disc.add(Dense(dense_layers[i],
            kernel_initializer=kernel_initializer,
            kernel_regularizer= l2(reg),
            bias_initializer = 'ones',
            bias_regularizer = l2(reg),
            activation = 'relu'
                       )
                 )
        disc.add(BatchNormalization())

    disc.add(Dense(num_classes, activation = 'softmax'))

    return disc

'''
DEFINE LOSS FUNCTIONS
'''

def discriminator_loss(real_output, fake_output):
    """
    from https://www.tensorflow.org/alpha/tutorials/generative/dcgan
    "This method quantifies how well the discriminator is able to distinguish real
    images from fakes. It compares the discriminator's predictions on real images
    to an array of 1s, and the discriminator's predictions on fake (generated) images
    to an array of 0s."
    """
    real_loss = binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    """
    from https://www.tensorflow.org/alpha/tutorials/generative/dcgan
    "The generator's loss quantifies how well it was able to trick the discriminator.
    Intuitively, if the generator is performing well, the discriminator will classify
    the fake images as real (or 1). Here, we will compare the discriminators decisions
    on the generated images to an array of 1s."
    """
    return binary_crossentropy(tf.ones_like(fake_output), fake_output)

def vae_loss(inputs, outputs, input_shape, z_mean, z_log_var):
    """
    Before training the generator as a GAN, we prime it by training it to reconstruct
    clean sensor readings.
    """
    # add loss and optimizer to vae
    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= input_shape[1]
    reconstruction_loss = K.sum(reconstruction_loss, axis=-1)

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    return K.mean(reconstruction_loss + kl_loss)
