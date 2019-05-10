"""
encode attack data w/ VAE and encourage separation
use VAE to generate synthetic attack data

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


#need to build encoder globally b/c of custom loss inputs
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


# hide old code that i'm scared to throw away and idk how to actually use verison control...
for _ in range(47):
    # build encoder
    #
    # conv[2] = input_shape[1]
    # inputs = Input(shape=input_shape,name='encoder_input')
    # conv_1 = Conv1D(filters = conv[0], kernel_size = K_SIZE,
    #          padding = PAD_TYPE, strides = 1)(inputs)
    # batch_1 = BatchNormalization(axis = 2)(conv_1)    # channels along last axis
    # pool_1 = MaxPooling1D(pool_size = RESOLUTION,padding = PAD_TYPE)(batch_1)
    # conv_2 = Conv1D(filters = conv[1], kernel_size = K_SIZE, padding = PAD_TYPE,
    #          strides = 1)(pool_1)
    # batch_2 = BatchNormalization(axis = 2)(conv_2)
    # pool_2 = MaxPooling1D(pool_size = RESOLUTION,padding = PAD_TYPE)(batch_2)
    # dense_1 = Dense(units = dense[0], activation = ACT_TYPE)(pool_2)
    # batch_3 = BatchNormalization(axis = 2)(dense_1)
    # dense_2 = Dense(units = dense[1], activation = ACT_TYPE)(batch_3)
    # batch_4 = BatchNormalization(axis = 2)(dense_2)
    # z_mean_un = Dense(units = LATENT_DIM)(batch_4)
    # z_log_var_un = Dense(units = LATENT_DIM, activation = ACT_TYPE)(batch_4)
    # z_mean = BatchNormalization(axis = 2)(z_mean_un)
    # z_log_var = BatchNormalization(axis = 2)(z_log_var_un)
    #
    # # use reparameterization trick to push the sampling out as input
    # # note that "output_shape" isn't necessary with the TensorFlow backend
    # # output shape is (batch_size, 6, LATENT_DIM)
    # z = Lambda(function = sampling, name='z')([z_mean, z_log_var])
    #
    # # build decoder branch
    # latent_inputs = Input(shape=K.int_shape(z)[1:], name='z_sampling')
    # d_dense_1 = Dense(units=dense[1], activation=ACT_TYPE)(latent_inputs)
    # d_batch_1 = BatchNormalization(axis=2)(d_dense_1)
    # d_dense_2 = Dense(units=dense[0], activation=ACT_TYPE)(d_batch_1)
    # d_batch_2 = BatchNormalization(axis=2)(d_dense_2)
    # d_dense_3 = Dense(units=dense[2], activation=ACT_TYPE)(d_batch_2)
    # d_batch_3 = BatchNormalization(axis=2)(d_dense_3)
    # d_up_1 = UpSampling1D(size=RESOLUTION)(d_batch_3)
    # d_conv_1 = Conv1D(filters=conv[0], kernel_size = K_SIZE,
    #            padding = PAD_TYPE, strides = 1)(d_up_1) #data_format = "channels_last"
    # d_batch_4 = BatchNormalization(axis=2)(d_conv_1)
    # d_up_2 = UpSampling1D(size=RESOLUTION)(d_batch_4)
    # d_mean_un = Conv1D(filters=conv[2], kernel_size = K_SIZE,
    #             padding = PAD_TYPE, strides = 1)(d_up_2) # data_format = "channels_last"
    # d_log_var_un = Conv1D(filters=conv[2], kernel_size = K_SIZE,
    #                padding = PAD_TYPE, strides = 1)(d_up_2) #data_format = "channels_last"
    # d_mean = BatchNormalization(axis = 2)(d_mean_un)
    # d_log_var = BatchNormalization(axis=2)(d_log_var_un)
    #
    # # join the branches by generating random normals. this is our reconstruction
    # x_outputs = Lambda(function = sampling, name='output')([d_mean, d_log_var])
    #
    # # build classification branch
    # c_conv_1 = Conv1D(filters = c_conv[0], kernel_size = K_SIZE,
    #          padding = PAD_TYPE, strides = 1)(latent_inputs)
    # c_batch_1 = BatchNormalization(axis = 2)(c_conv_1)    # channels along last axis
    # c_pool_1 = MaxPooling1D(pool_size = RESOLUTION,padding = PAD_TYPE)(c_batch_1)
    # c_conv_2 = Conv1D(filters = c_conv[1], kernel_size = K_SIZE,
    #          padding = PAD_TYPE, strides = 1)(c_pool_1)
    # c_batch_2 = BatchNormalization(axis = 2)(c_conv_2)    # channels along last axis
    # c_pool_2 = MaxPooling1D(pool_size = RESOLUTION,padding = PAD_TYPE)(c_batch_2)
    # c_flat = Flatten()(c_pool_2)
    # c_dense_1 = Dense(units = c_dense[0], activation = ACT_TYPE)(c_flat)
    # c_batch_1 = BatchNormalization(axis = 1)(c_dense_1)
    # c_dense_2 = Dense(units = c_dense[1], activation = ACT_TYPE)(c_batch_1)
    # c_batch_2 = BatchNormalization(axis = 1)(c_dense_2)
    # y_outputs = Dense(units = NUM_CLASSES, activation = ACT_TYPE)(c_batch_2)
    #
    # # instantiate subsidiary models
    # encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    # decoder = Model(latent_inputs, [d_mean,d_log_var,x_outputs], name='decoder')
    # classifier = Model(latent_inputs, y_outputs, name='classifier')


'''
TRAIN
'''

'''
VISUALIZE
'''
def get_prediction(vae_pred):
    '''
    input: unformatted VAE prediction
    return: y_pred
    '''
    return (vae_pred[1][:,1]>=0.5)+0

def viz(want_to_see, window, loss_df):
    '''
    visualize: periodic reconstructions of sensor readings
    inputs
    #   want_to_see - show plots in real time? otherwise just write to disc
    '''
    lb, ub = np.min(window), np.max(window)

    fig, ax = plt.subplots()
    sns.heatmap(window[0],vmin=lb, vmax=ub) # crashes ..
    plt.savefig(os.path.join(im_dir,'original_window.png'))

    # look at windows
    for n in np.arange(1,num_times+1):

        fig, ax = plt.subplots()
        window_pred=window[n]
        sns.heatmap(window_pred,vmin=lb, vmax=ub)
        plt.savefig(os.path.join(im_dir,'reconstructed_at_ep_{:04d}.png'.format(n*ep_at_a_time)))

    # look at loss
    fig, ax = plt.subplots()
    ax.plot(loss_df['loss'], '-b', label='Training loss')
    ax.plot(loss_df['val_loss'], '--r', label='Validation loss')
    ax.legend(loc='upper right', frameon=False)
    plt.savefig(os.path.join(dir,'loss.png'))

    # classification and generation loss separately
    fig, ax = plt.subplots()
    ax.plot(loss_df['loss'], '-b', label='Training loss')
    ax.plot(loss_df['val_loss'], '--r', label='Validation loss')
    ax.legend(loc='upper right', frameon=False)
    plt.savefig(os.path.join(dir,'loss.png'))

    plt.show() if want_to_see


'''
TRAIN
'''

def train_vae(output_dir,
              gen_weight,
              checkins,
              epochs,
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
    # set up directories
    image_dir=os.path.join(dir,'viz')
    os.makedirs(dir,exist_ok=True)
    os.makedirs(image_dir,exist_ok=True)

    # instantiate
    vae = Model(inputs, [x_activated,y_activated], name='vae')

    # model components
    losses={'generation': vae_loss(z_mean,z_log_var,inputs,x_activated,input_shape),'classification': class_loss}
    loss_weights={'generation': gen_weight,'classification': 1.0}
    opt=sgd(lr=0.01,clipnorm=1.)
    vae.compile(optimizer=opt,loss=losses,loss_weights=loss_weights)
    w=class_weight.compute_class_weight('balanced',np.unique(y_tr),y_tr)

    # grab first attack window
    window=X_val[np.where(y_val==1)[0][0]+12]
    (w1,w2)=window.shape
    window=window.reshape(1,w1,w2)

    # train and periodically check in
    for n in np.arange(checkins):

        vae.fit(X_tr,{'generation':y_tr,'classification':y_tr},
                epochs=epochs, batch_size=batch_size,
                class_weight={'generation':w,'classification':w},
                validation_data=(X_val,{'generation':y_val,'classification':y_val}))

        # save visual progress
        window_pred=vae.predict(window[0].reshape(1,x,y))[0]
        window=np.concatenate((window,window_pred),axis=0)

        # save loss
        d=vae.history.history
        history=np.stack([d['loss'],d['val_loss'],
                          d['classification_loss'],d['val_classification_loss'],
                          d['generation_loss'],d['val_generation_loss']]).T
        loss_history = (history if n==0 else np.concatenate((loss_history,history),axis=0))

    ##### save everything
    vae.save_weights(os.path.join(dir,'vae.h5'))
    encoder.save_weights(os.path.join(dir,'encoder.h5'))
    decoder.save_weights(os.path.join(dir,'decoder.h5'))

    np.save(os.path.join(dir,'window.npy'),window)

    loss_df = pd.DataFrame()
    loss_df['loss'],loss_df['val_loss']=loss_history[:,0],loss_history[:,1]
    loss_df['classification_loss'],loss_df['val_classification_loss']=loss_history[:,2],loss_history[:,3]
    loss_df['generation_loss'],loss_df['val_generation_loss']=loss_history[:,4],loss_history[:,5]

    loss_df.to_csv(os.path.join(dir,'loss_results.csv'), index=False)

    # visualize windows and loss
    viz(want_to_see, window, loss_df)

if __name__ == '__main__':
    '''
    train VAE w/ weighted MSE
    '''
