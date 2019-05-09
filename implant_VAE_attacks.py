"""
encode attack data w/ VAE and encourage separation
use VAE to generate synthetic attack data

multiple losses: https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
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

'''
CONSTANTS
'''

TEST_SIZE=0.3
input_shape=(24,36)
k_size=5
conv=[60,80,36]
resolution=2
dense=[200,160,480]
c_conv=[50,30,10]
c_dense=[10]
latent_dim=100
pad_type='same'
act_type='relu'
ep_at_a_time=10
num_times=3
num_classes=2
b_s=128
gen_weight=2.0      # how much more to weight generation acc. vs. class. acc.


'''
DATA
'''

(_, _), (X, y), names = format_data.get_rolled_data()
X_tr, X_val, y_tr, y_val = train_test_split(X, y, stratify=y, test_size=TEST_SIZE, shuffle=True)

'''
VISUALIZE
'''


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

def custom_loss(z_mean, z_log_var, inputs, outputs):
    """
    define custom loss based on VAE loss and MSE
    with hopes that classes will be encoded separately in order for vae
    to successfully classify examples
    """
    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= inputs.shape[1]
    reconstruction_loss = K.sum(reconstruction_loss,axis=-1)

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    v_loss = K.mean(reconstruction_loss+kl_loss)

    def loss(y_true, y_pred):
        ones = K.ones_like(y_true[0,:])
        idx = K.cumsum(ones) # weighted
        return K.mean((1/idx)*K.square(y_true-y_pred)) + v_loss
        #return K.mean(K.square(y_pred-y_true)) + v_loss

    # return a function
    return loss

def vae_loss(z_mean, z_log_var, inputs, outputs):
    '''
    loss for data generation branch
    '''
    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= inputs.shape[1]
    reconstruction_loss = K.sum(reconstruction_loss,axis=-1)

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    return K.mean(reconstruction_loss+kl_loss)

# losses
losses = {"generation_output": vae_loss(z_mean,z_log_var,inputs,outputs),
	      "classification_output": "mse"}
loss_weights = {"generation_output": gen_weight,
                "classification_output": 1.0}


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

# build decoder branch
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
x_outputs = Lambda(function = sampling, name='output')([d_mean, d_log_var])

# build classification branch
c_conv_1 = Conv1D(filters = c_conv[0], kernel_size = k_size,
         padding = pad_type, strides = 1)(latent_inputs)
c_batch_1 = BatchNormalization(axis = 2)(c_conv_1)    # channels along last axis
c_pool_1 = MaxPooling1D(pool_size = resolution,padding = pad_type)(c_batch_1)
c_conv_2 = Conv1D(filters = c_conv[1], kernel_size = k_size, padding = pad_type,
         strides = 1)(c_pool_1)
c_batch_2 = BatchNormalization(axis = 2)(c_conv_2)
c_pool_2 = MaxPooling1D(pool_size = resolution,padding = pad_type)(c_batch_2)
c_flat = Flatten()(c_pool_2)
c_dense_1 = Dense(units = c_dense[0], activation = act_type)(c_flat)
c_batch_3 = BatchNormalization(axis = 1)(c_dense_1)
y_outputs = Dense(units = num_classes, activation = act_type)(c_batch_3)

# instantiate models
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
decoder = Model(latent_inputs, [d_mean,d_log_var,x_outputs], name='decoder')
classifier = Model(latent_inputs, y_outputs, name='classifier')
x_outputs = decoder(encoder(inputs)[2])[2]
y_outputs = classifier(encoder(inputs)[2])
vae = Model(inputs, [x_outputs,y_outputs], name='vae')

vae.compile(optimizer='adam',
            loss=custom_loss(z_mean, z_log_var, inputs, x_outputs),
            metrics=['acc'])




if __name__ == '__main__':
    '''
    train VAE w/ weighted MSE
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

    # grab an attack window
    first_attack_ind=np.where(y_val==1)[0][0]
    window=X_val[first_attack_ind+12]
    (x,y) = window.shape
    window=window.reshape(1,x,y)

    # train
    for n in np.arange(num_times):
        vae.fit(X_tr,y_tr,epochs=ep_at_a_time,batch_size=b_s,validation_data=(X_val,y_val))

        # save visual progress
        window_pred=vae.predict(window[0].reshape(1,x,y))
        window=np.concatenate((window,window_pred),axis=0)

        # save loss
        history=np.stack([vae.history.history['loss'],vae.history.history['val_loss'],
                          vae.history.history['acc'],vae.history.history['val_acc']]).T
        loss_history = (history if n==0 else np.concatenate((loss_history,history),axis=0))
    # get the images
    viz()

    # save weights, loss, windows
    vae.save_weights(os.path.join(dir,'vae.h5'))
    encoder.save_weights(os.path.join(dir,'encoder.h5'))
    decoder.save_weights(os.path.join(dir,'decoder.h5'))
    np.save(os.path.join(dir,'window.npy'),window)
    loss_df = pd.DataFrame()
    loss_df['loss'],loss_df['val_loss']=loss_history[:,0],loss_history[:,1]
    loss_df['acc'],loss_df['val_acc']=loss_history[:,2],loss_history[:,3]
    loss_df.to_csv(os.path.join(dir,'loss_results.csv'), index=False)
