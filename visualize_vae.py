"""
visualize training process of vae
does generated data look like it should?
"""
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

from tensorflow.keras.layers import Input, Lambda, Dense, BatchNormalization, Activation
from tensorflow.keras.layers import Dropout, Conv1D, Flatten, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse, categorical_crossentropy, binary_crossentropy
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
#from tensorflow.keras.optimizers import sgd
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
import attack_GAN

'''
CONSTANTS
'''


input_shape=(24,36)
k_size=5
conv=[60,80,36]
resolution=2
dense=[200,160,480]
latent_dim=100
pad_type='same'
act_type='relu'
ep_at_a_time=50
num_times=4
b_s=128
TEST_SIZE=0.3


'''
DATA
'''

(clean, y_clean), (_, _), names = format_data.get_rolled_data()
clean_train, clean_val = train_test_split(clean, test_size=TEST_SIZE, shuffle=True)

'''
MODEL
'''

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

reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= input_shape[1]
reconstruction_loss = K.sum(reconstruction_loss, axis=-1)

kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5

v_loss = K.mean(reconstruction_loss + kl_loss)

#v_loss = attack_GAN.vae_loss(inputs, outputs, input_shape, z_mean, z_log_var)
vae.add_loss(v_loss)
vae.compile(optimizer='adam')

'''
VISUALIZE
'''
def viz():

    os.makedirs(dir, exist_ok=True)

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
    ax.plot(loss_history[:,0], '-b', label='Training loss')
    ax.plot(loss_history[:,1], '--r', label='Validation loss')
    ax.legend(loc='upper right', frameon=False)
    plt.savefig(os.path.join(dir,'loss.png'))

    plt.show()

if __name__ == '__main__':
    '''
    prime the generator
    '''
    parser = argparse.ArgumentParser()
    help_ = "give directory name"
    parser.add_argument("-d", "--directory", help=help_)
    args = parser.parse_args()

    # get output directory
    dir = args.directory if args.directory else "temp"
    im_dir=os.path.join(dir,'viz')
    os.makedirs(dir,exist_ok=True)
    os.makedirs(im_dir, exist_ok=True)

    # grab a window
    window=clean_val[0]
    (x,y) = window.shape
    window=window.reshape(1,x,y)

    # train
    for n in np.arange(num_times):
        vae.fit(clean_train,epochs=ep_at_a_time,batch_size=b_s,validation_data=(clean_val,None))

        # save visual progress
        window_pred=vae.predict(window[0].reshape(1,x,y))
        window=np.concatenate((window,window_pred),axis=0)

        # save loss
        history=np.stack([vae.history.history['loss'],vae.history.history['val_loss']]).T
        loss_history = (history if n==0 else np.concatenate((loss_history,history),axis=0))
    # get the images
    viz()

    # save weights, loss, windows
    vae.save_weights(os.path.join(dir,'vae.h5'))
    decoder.save_weights(os.path.join(dir,'decoder.h5'))
    np.save(os.path.join(dir,'window.npy'),window)
    loss_df = pd.DataFrame()
    loss_df['loss'],loss_df['val_loss']=loss_history[:,0],loss_history[:,1]
    loss_df.to_csv(os.path.join(dir,'loss_results.csv'), index=False)
