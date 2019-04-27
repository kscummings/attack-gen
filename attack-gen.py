'''
final 6.883 project
continuation of work done at ORNL w Jason Laska
trying to build a GAN instead of manually generating attacks
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.metrics import confusion_matrix
from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from keras.layers import Dense, BatchNormalization, Activation, Dropout, Conv1D, Flatten, MaxPool1D
from keras.models import Model, Sequential
from keras.losses import categorical_crossentropy, binary_crossentropy
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


def gen_model(num_dense=3,
              input_shape=(24,36),
              reg=.00,
              kernel_initializer='glorot_uniform'):
    """
    Build an attack generator that takes clean data window as input
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
