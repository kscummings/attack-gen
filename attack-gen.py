'''
final 6.883 project
continuation of work done at ORNL w Jason Laska
trying to build a GAN instead of manually generating attacks
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

from keras.layers import Input, Lambda, Dense, BatchNormalization, Activation, Dropout, Conv1D, Flatten, MaxPooling1D, UpSampling1D
from keras.models import Model, Sequential
from keras.losses import mse, categorical_crossentropy, binary_crossentropy
from keras.utils import plot_model, to_categorical
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import sgd
from keras.callbacks import EarlyStopping

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

'''
BUILD GENERATOR
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


def vae_gen_model(input_shape=(24,36),
                  k_size=5,
                  conv=[60,80,36],
                  resolution=2,
                  dense=[200,160,480],
                  latent_dim=100,
                  pad_type='same',
                  act_type='relu'):
    """
    Build attack generator that takes clean data window as input
    Architecture based on Chandy's VAE
    """

    # build encoder

    conv[3] = input_shape[1]
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

    # instantiate encoder
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

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

    # instantiate decoder
    decoder = Model(latent_inputs, [d_mean,d_log_var,outputs], name='decoder')

    # instantiate vae
    outputs = decoder(encoder(inputs)[2])[2]
    vae = Model(inputs, outputs, name='vae')



def gen_model(num_dense=3,
              input_shape=(24,36),
              reg=.00,
              kernel_initializer='glorot_uniform'):
    """
    Build a dumb attack generator that takes clean data window as input
    For now, just some dense layers...
    # Arguments
        num_dense = number of modules in network
        input_shape = input dim of each obs
        reg = regularization parameter
        kernel_initializer = NN parameter initialization or something like that idk

    """
    gen=Sequential()
    dense_dim=input_shape[1]

    # i think this flattens automatically
    gen.add(Dense(dense_dim,
                  input_shape=input_shape,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer = l2(reg),
                  bias_initializer = 'ones',
                  bias_regularizer = l2(reg),
                  activation = 'relu'))


    for i in range(num_dense-1):
        gen.add(BatchNormalization())
        gen.add(Dense(dense_dim,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer= l2(reg),
                      bias_initializer = 'ones',
                      bias_regularizer = l2(reg),
                      activation = 'relu'))

    gen.add(BatchNormalization())
    gen.add(Dense(dense_dim,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer= l2(reg),
                    bias_initializer = 'ones',
                    bias_regularizer = l2(reg),
                    activation = 'softmax'))

    return gen

'''
BUILD DISCRIMINATOR
'''

def disc_model(conv_layers = [50,50,50],
                 dense_layers = [10,10],
                 k_size = 5,
                 pad_type = 'same',
                 resolution = 2,
                 kernel_initializer = 'glorot_uniform',
                 num_classes = 2,
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
    disc.add(MaxPool1D(pool_size = resolution, padding = pad_type))
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
        disc.add(MaxPool1D(pool_size = resolution, padding = pad_type))
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

'''
IMPORT AND PREPROCESS DATA
'''

W_LENGTH = 24       # time window of each rolled observation
STD_PCT = 0.1       # percentage of features to toss, determined by lowest variance

# import raw clean data
train = pd.read_csv("./batadal/BATADAL_dataset03.csv")
y_train = train['ATT_FLAG'].values
train = train.drop(columns=['DATETIME','ATT_FLAG'])
names = train.columns
train = train.values            # keras takes np arrays

# raw train data with real cyber attacks and re-labeled attack flags
real_train = pd.read_csv("./batadal/BATADAL_dataset04_manualflags.csv")
y_real_train = real_train['ATT_FLAG'].values
real_train = real_train.drop(columns=['DATETIME','ATT_FLAG']).values

# standardize and shuck
scaler = MinMaxScaler()
train = scaler.fit_transform(train)
real_train = scaler.transform(real_train)
keep = train.std(0) > np.percentile(train.std(0), STD_PCT)
train = train[:,keep]
real_train = real_train[:,keep]
names = names[keep]

# roll real train data
NUM_OBS = real_train.shape[0] - W_LENGTH + 1
real_rolled = np.zeros((NUM_OBS, W_LENGTH, len(names)))
y_real_rolled = np.zeros(NUM_OBS)
for w in range(NUM_OBS):
    real_rolled[w] = real_train[w:(w+W_LENGTH)]
    y_real_rolled[w] = np.any(y_real_train[w:(w+W_LENGTH)]==1).astype(int)
