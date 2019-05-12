"""
encode attack data w/ VAE and encourage class separation in the encoding
use VAE to generate labeled synthetic attack data

multiple losses: https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.stats import multivariate_normal

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle, class_weight
from sklearn.model_selection import GridSearchCV,train_test_split

from keras.layers import Input, Lambda, Dense, BatchNormalization, Activation
from keras.layers import Dropout, Conv1D, Flatten, MaxPooling1D, UpSampling1D
from keras.models import Model, Sequential
from keras.losses import mse, sparse_categorical_crossentropy, binary_crossentropy
from keras.utils import plot_model, to_categorical
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import sgd
#from tensorflow.keras.callbacks import EarlyStopping

import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
# uncomment if in virtualenv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import os

import format_data

'''
CONSTANTS
'''

# encoder params
input_shape=(24,36)
k_size=5
conv=[60,80,36]
resolution=2
dense=[200,160,480]
latent_dim=100
pad_type='same'
act_type='relu'

# training params
TEST_SIZE=0.3
EPOCHS=10       # num epochs at a time
NUM_TIMES=3
NUM_CLASSES=2   # bxe with 1 class wasn't training properly
BATCH_SIZE=256

GEN_WEIGHT=2.0      # how much more to weight generation acc. vs. class. acc.
LOSS='sparse_categorical_crossentropy' #mse, binary_crossentropy (bxe isn't working...)

'''
DATA
'''

(_, _), (X, y), names = format_data.get_rolled_data()
X_tr, X_val, y_tr, y_val = train_test_split(X, y, stratify=y, test_size=TEST_SIZE, shuffle=True)

'''
MODEL
'''

def sampling(args):
    '''
    Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent array of size .... um
    '''

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim1 = K.int_shape(z_mean)[1]
    dim2 = K.int_shape(z_mean)[2]
    # by default, random_normal has mean=0 and sd=1.0
    epsilon = K.random_normal(shape=(batch, dim1, dim2))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon # latent tensor


def vae_loss(z_mean, z_log_var, inputs, outputs, input_shape):
    '''
    custom loss for data generation branch
    '''
    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= input_shape[1]
    reconstruction_loss = K.sum(reconstruction_loss,axis=-1)

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    def loss(y_true,y_pred):
        return K.mean(reconstruction_loss+kl_loss)

    return loss


def build_encoder(input_layer,
                  k_size=5,
                  conv=[60,80,36],
                  resolution=2,
                  dense=[200,160,480],
                  latent_dim=100,
                  pad_type='same',
                  act_type='relu'):
    '''
    can't actually use this, b/c need access to intermediate layers
    for custom loss function
    '''

    #conv[2] = input_shape[1]
    conv1 = Conv1D(filters = conv[0], kernel_size = k_size,
             padding = pad_type, strides = 1)(input_layer)
    batch1 = BatchNormalization(axis = 2)(conv1)    # channels along last axis
    pool1 = MaxPooling1D(pool_size = resolution,padding = pad_type)(batch1)
    conv2 = Conv1D(filters = conv[1], kernel_size = k_size, padding = pad_type,
             strides = 1)(pool1)
    batch2 = BatchNormalization(axis = 2)(conv2)
    pool2 = MaxPooling1D(pool_size = resolution,padding = pad_type)(batch2)
    dense1 = Dense(units = dense[0], activation = act_type)(pool2)
    batch3 = BatchNormalization(axis = 2)(dense1)
    dense2 = Dense(units = dense[1], activation = act_type)(batch3)
    batch4 = BatchNormalization(axis = 2)(dense2)

    # mean and log-variance branches
    z_mean_un = Dense(units = latent_dim)(batch_4)
    z_mean = BatchNormalization(axis = 2)(z_mean_un)
    z_log_var_un = Dense(units = latent_dim, activation = act_type)(batch_4)
    z_log_var = BatchNormalization(axis = 2)(z_log_var_un)

    # use reparameterization trick to push the sampling out as input
    z = Lambda(function = sampling, name='z')([z_mean, z_log_var])

    # encoder
    return Model(input_layer, [z_mean, z_log_var, z], name='encoder')


def build_decoder(latent_input_shape=(6,100),
                  k_size=5,
                  conv=[60,36],
                  resolution=2,
                  dense=[160,200,480],
                  latent_dim=100,
                  pad_type='same',
                  act_type='relu'):
    '''
    VAE branch
    '''

    latent_inputs = Input(shape=latent_input_shape, name='z_sampling')
    dense1 = Dense(units=dense[0], activation=act_type)(latent_inputs)
    batch1 = BatchNormalization(axis=2)(dense1)
    dense2 = Dense(units=dense[1], activation=act_type)(batch1)
    batch2 = BatchNormalization(axis=2)(dense2)
    dense3 = Dense(units=dense[2], activation=act_type)(batch2)
    batch3 = BatchNormalization(axis=2)(dense3)
    up1 = UpSampling1D(size=resolution)(batch3)
    conv1 = Conv1D(filters=conv[0], kernel_size = k_size,
               padding = pad_type, strides = 1)(up1) #data_format = "channels_last"
    batch4 = BatchNormalization(axis=2)(conv1)
    up2 = UpSampling1D(size=resolution)(batch4)

    # mean/log-variance
    mean_un = Conv1D(filters=conv[1], kernel_size = k_size,
                padding = pad_type, strides = 1)(up2) # data_format = "channels_last"
    mean = BatchNormalization(axis=2)(mean_un)

    log_var_un = Conv1D(filters=conv[1], kernel_size = k_size,
                   padding = pad_type, strides = 1)(up2) #data_format = "channels_last"
    log_var = BatchNormalization(axis=2)(log_var_un)

    # join the branches by generating random normals. this is our reconstruction
    x_outputs = Lambda(function = sampling, name='output')([mean, log_var])

    # decoder
    return Model(latent_inputs, [mean,log_var,x_outputs], name='decoder')

def build_classifier(latent_input_shape=(6,100),
                     conv=[70,30],
                     dense=[100,50],
                     act_type='relu',
                     pad_type='same',
                     num_classes=2,
                     resolution=2,
                     k_size=5):
    '''
    prediction branch
    '''

    latent_inputs = Input(shape=latent_input_shape, name='z_sampling')
    conv1 = Conv1D(filters = conv[0], kernel_size = k_size, padding = pad_type, strides = 1)(latent_inputs)
    batch1 = BatchNormalization(axis = 2)(conv1)    # channels along last axis
    pool1 = MaxPooling1D(pool_size = resolution,padding = pad_type)(batch1)
    conv2 = Conv1D(filters = conv[1], kernel_size = k_size, padding = pad_type, strides = 1)(pool1)
    batch2 = BatchNormalization(axis = 2)(conv2)    # channels along last axis
    pool2 = MaxPooling1D(pool_size = resolution,padding = pad_type)(batch2)
    flat = Flatten()(pool2)
    dense1 = Dense(units = dense[0], activation = act_type)(flat)
    batch1 = BatchNormalization(axis = 1)(dense1)
    dense2 = Dense(units = dense[1], activation = act_type)(batch1)
    batch2 = BatchNormalization(axis = 1)(dense2)
    y_outputs = Dense(units = num_classes, activation = act_type)(batch2)

    # classifier
    return Model(latent_inputs, y_outputs, name='classifier')

def get_vae():
    '''
    get model architecture so we can load weights in format_data
    '''
    inputs=Input(shape=input_shape,name='encoder_input')
    encoder=build_encoder(inputs)
    decoder=build_decoder()
    classifier=build_classifier()
    x_outputs = decoder(encoder(inputs)[2])[2]
    x_activated = Activation('linear',name='generation')(x_outputs)
    y_outputs = classifier(encoder(inputs)[2])
    y_activated = Activation('softmax',name='classification')(y_outputs)
    return Model(inputs, [x_activated,y_activated], name='vae')


#need to build encoder globally so custom loss can access inputs
inputs = Input(shape=input_shape,name='encoder_input')
conv1 = Conv1D(filters = conv[0], kernel_size = k_size,
         padding = pad_type, strides = 1)(inputs)
batch1 = BatchNormalization(axis = 2)(conv1)    # channels along last axis
pool1 = MaxPooling1D(pool_size = resolution,padding = pad_type)(batch1)
conv2 = Conv1D(filters = conv[1], kernel_size = k_size, padding = pad_type,
         strides = 1)(pool1)
batch2 = BatchNormalization(axis = 2)(conv2)
pool2 = MaxPooling1D(pool_size = resolution,padding = pad_type)(batch2)
dense1 = Dense(units = dense[0], activation = act_type)(pool2)
batch3 = BatchNormalization(axis = 2)(dense1)
dense2 = Dense(units = dense[1], activation = act_type)(batch3)
batch4 = BatchNormalization(axis = 2)(dense2)

# mean and log-variance branches
z_mean_un = Dense(units = latent_dim)(batch4)
z_mean = BatchNormalization(axis = 2)(z_mean_un)
z_log_var_un = Dense(units = latent_dim, activation = act_type)(batch4)
z_log_var = BatchNormalization(axis = 2)(z_log_var_un)

# use reparameterization trick to push the sampling out as input
z = Lambda(function = sampling, name='z')([z_mean, z_log_var])


# construct VAE
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
decoder = build_decoder()
classifier = build_classifier()
x_outputs = decoder(encoder(inputs)[2])[2]
x_activated = Activation('linear',name='generation')(x_outputs) # need to name output layers
y_outputs = classifier(encoder(inputs)[2])
y_activated = Activation('softmax',name='classification')(y_outputs)


'''
VISUALIZE
'''
def get_prediction(vae_pred):
    '''
    input: unformatted VAE prediction
    return: y_pred
    '''
    return (vae_pred[1][:,1]>=0.5)+0

def viz(output_dir,
        want_to_see,
        window,
        loss_df,
        checkins,
        epochs,
        see_from=0):
    '''
    visualize: periodic reconstructions of sensor readings
    inputs
    #   output_dir - directory to save files
    #   loss_dir - subdirectory to save loss plots
    #   want_to_see - show plots in real time? otherwise just write to disc
    #   window - set of windows to visualize, collected from train fcn
    #   loss_df - loss results
    #   see_from - first observation to visualize in loss
    '''
    # set up subsidiary directories
    loss_dir=os.path.join(output_dir,"loss_plots")
    im_dir=os.path.join(output_dir,"window_heatmaps")
    os.makedirs(im_dir,exist_ok=True)
    os.makedirs(loss_dir,exist_ok=True)

    # first heatmap
    lb, ub = np.min(window), np.max(window)
    fig, ax = plt.subplots()
    sns.heatmap(window[0],vmin=lb, vmax=ub) # crashes ..
    plt.savefig(os.path.join(im_dir,'original_window.png'))

    # look at windows
    for n in np.arange(checkins+1):
        fig, ax = plt.subplots()
        window_pred=window[n+1]
        sns.heatmap(window_pred,vmin=lb, vmax=ub)
        plt.savefig(os.path.join(im_dir,'reconstructed_at_ep_{:04d}.png'.format(n*epochs)))

    # look at loss
    fig, ax = plt.subplots()
    ax.plot(loss_df['loss'][see_from:], '-b', label='Training loss')
    ax.plot(loss_df['val_loss'][see_from:], '--r', label='Validation loss')
    ax.legend(loc='upper right', frameon=False)
    plt.savefig(os.path.join(loss_dir,'loss.png'))

    # generation loss
    fig, ax = plt.subplots()
    ax.plot(loss_df['generation_loss'][see_from:], '-b', label='Training loss')
    ax.plot(loss_df['val_generation_loss'][see_from:], '--r', label='Validation loss')
    ax.legend(loc='upper right', frameon=False)
    plt.title('Generation loss')
    plt.savefig(os.path.join(loss_dir,'generation_loss.png'))

    # classification loss
    fig, ax = plt.subplots()
    ax.plot(loss_df['classification_loss'][see_from:], '-b', label='Training loss')
    ax.plot(loss_df['val_classification_loss'][see_from:], '--r', label='Validation loss')
    ax.legend(loc='upper right', frameon=False)
    plt.title('Classification loss')
    plt.savefig(os.path.join(loss_dir,'classification_loss.png'))

    plt.show() if want_to_see else True


'''
TRAIN
'''

def train_vae(data,
              output_dir,
              gen_weight,
              checkins,
              epochs,
              classification_weight=1,
              see_from=10,
              want_to_see=False,
              batch_size=256,
              class_loss='sparse_categorical_crossentropy'):
    '''
    train the VAE, save visualizations and
    inputs
    #   output_dir - directory to write results
    #   gen_weight - weight of generation loss relative to classificaiton loss weight (set to 1.0)
    #   checkins - num. sensor reading windows to visualize during training process
    #   epochs - num. epochs per checkin
    #   want_to_see - show plots; otherwise, just write to disc
    #   batch_size - num. obs. to train at a time
    #   class_loss - loss to use for classification branch
    '''
    os.makedirs(output_dir,exist_ok=True)
    (X_tr,y_tr,X_val,y_val)=data

    # instantiate
    vae = Model(inputs, [x_activated,y_activated], name='vae')

    # model components
    losses={'generation': vae_loss(z_mean,z_log_var,inputs,x_activated,input_shape),'classification': class_loss}
    loss_weights={'generation': gen_weight,'classification': classification_weight}
    opt=sgd(lr=0.01,clipnorm=1.)
    vae.compile(optimizer=opt,loss=losses,loss_weights=loss_weights)
    w=class_weight.compute_class_weight('balanced',np.unique(y_tr),y_tr)

    # grab first attack window
    window=X_val[np.where(y_val==1)[0][0]+12]
    (w1,w2)=window.shape
    window=window.reshape(1,w1,w2)

    # look at initial prediction, pre-training
    window_pred=vae.predict(window[0].reshape(1,w1,w2))[0]
    window=np.concatenate((window,window_pred),axis=0)

    # train and periodically check in
    for n in np.arange(checkins):

        vae.fit(X_tr,{'generation':y_tr,'classification':y_tr},
                epochs=epochs, batch_size=batch_size,
                class_weight={'generation':w,'classification':w},
                validation_data=(X_val,{'generation':y_val,'classification':y_val}))

        # save visual progress
        window_pred=vae.predict(window[0].reshape(1,w1,w2))[0]
        window=np.concatenate((window,window_pred),axis=0)

        # save loss
        d=vae.history.history
        history=np.stack([d['loss'],d['val_loss'],
                          d['classification_loss'],d['val_classification_loss'],
                          d['generation_loss'],d['val_generation_loss']]).T
        loss_history = (history if n==0 else np.concatenate((loss_history,history),axis=0))

    ##### save everything
    vae.save_weights(os.path.join(output_dir,'vae.h5'))
    encoder.save_weights(os.path.join(output_dir,'encoder.h5'))
    decoder.save_weights(os.path.join(output_dir,'decoder.h5'))
    classifier.save_weights(os.path.join(output_dir,'classifier.h5'))

    np.save(os.path.join(output_dir,'window.npy'),window)

    loss_df = pd.DataFrame(loss_history,columns=['loss','val_loss','classification_loss',
                                'val_classification_loss','generation_loss','val_generation_loss'])
    loss_df.to_csv(os.path.join(output_dir,'loss_results.csv'), index=False)

    # visualize windows and loss
    viz(output_dir,want_to_see,window,loss_df,checkins,epochs,see_from)

if __name__ == '__main__':
    '''
    train VAE
    '''
    # does this script work? verdict: yes! :)
    train_vae(data=(X_tr,y_tr,X_val,y_val),
              output_dir="test",
              gen_weight=GEN_WEIGHT,
              checkins=NUM_TIMES,
              epochs=EPOCHS,
              see_from=10,
              want_to_see=True,
              batch_size=BATCH_SIZE,
              class_loss='sparse_categorical_crossentropy')
