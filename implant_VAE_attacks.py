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
from sklearn.utils import shuffle, class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

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

TEST_SIZE=0.3
input_shape=(24,36)
K_SIZE=5
conv=[60,80,36]
RESOLUTION=2
dense=[200,160,480]
c_conv=[70,30]
c_dense=[100,50]
LATENT_DIM=100
PAD_TYPE='same'
ACT_TYPE='relu'
EPOCHS=10       #num epochs at a time
NUM_TIMES=3
NUM_CLASSES=2
BATCH_SIZE=128
GEN_WEIGHT=2.0      # how much more to weight generation acc. vs. class. acc.
LOSS='sparse_categorical_crossentropy' #mse, binary_crossentropy

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

def custom_loss(z_mean, z_log_var, inputs, outputs, input_shape):
    """
    define custom loss based on VAE loss and MSE
    with hopes that classes will be encoded separately in order for vae
    to successfully classify examples
    """
    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= input_shape[1]
    reconstruction_loss = K.sum(reconstruction_loss,axis=-1)

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    v_loss = K.mean(reconstruction_loss+kl_loss)

    def loss(y_true, y_pred):
        ones = K.ones_like(y_true[0,:])
        idx = K.cumsum(ones) # weighted
        return K.mean((1/idx)*K.square(y_true-y_pred)) + 2*v_loss
        #return K.mean(K.square(y_pred-y_true)) + v_loss

    # return a function
    return loss

def vae_loss(z_mean, z_log_var, inputs, outputs, input_shape):
    '''
    loss for data generation branch
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




# build encoder

conv[2] = input_shape[1]
inputs = Input(shape=input_shape,name='encoder_input')
conv_1 = Conv1D(filters = conv[0], kernel_size = K_SIZE,
         padding = PAD_TYPE, strides = 1)(inputs)
batch_1 = BatchNormalization(axis = 2)(conv_1)    # channels along last axis
pool_1 = MaxPooling1D(pool_size = RESOLUTION,padding = PAD_TYPE)(batch_1)
conv_2 = Conv1D(filters = conv[1], kernel_size = K_SIZE, padding = PAD_TYPE,
         strides = 1)(pool_1)
batch_2 = BatchNormalization(axis = 2)(conv_2)
pool_2 = MaxPooling1D(pool_size = RESOLUTION,padding = PAD_TYPE)(batch_2)
dense_1 = Dense(units = dense[0], activation = ACT_TYPE)(pool_2)
batch_3 = BatchNormalization(axis = 2)(dense_1)
dense_2 = Dense(units = dense[1], activation = ACT_TYPE)(batch_3)
batch_4 = BatchNormalization(axis = 2)(dense_2)
z_mean_un = Dense(units = LATENT_DIM)(batch_4)
z_log_var_un = Dense(units = LATENT_DIM, activation = ACT_TYPE)(batch_4)
z_mean = BatchNormalization(axis = 2)(z_mean_un)
z_log_var = BatchNormalization(axis = 2)(z_log_var_un)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
# output shape is (batch_size, 6, LATENT_DIM)
z = Lambda(function = sampling, name='z')([z_mean, z_log_var])

# build decoder branch
latent_inputs = Input(shape=K.int_shape(z)[1:], name='z_sampling')
d_dense_1 = Dense(units=dense[1], activation=ACT_TYPE)(latent_inputs)
d_batch_1 = BatchNormalization(axis=2)(d_dense_1)
d_dense_2 = Dense(units=dense[0], activation=ACT_TYPE)(d_batch_1)
d_batch_2 = BatchNormalization(axis=2)(d_dense_2)
d_dense_3 = Dense(units=dense[2], activation=ACT_TYPE)(d_batch_2)
d_batch_3 = BatchNormalization(axis=2)(d_dense_3)
d_up_1 = UpSampling1D(size=RESOLUTION)(d_batch_3)
d_conv_1 = Conv1D(filters=conv[0], kernel_size = K_SIZE,
           padding = PAD_TYPE, strides = 1)(d_up_1) #data_format = "channels_last"
d_batch_4 = BatchNormalization(axis=2)(d_conv_1)
d_up_2 = UpSampling1D(size=RESOLUTION)(d_batch_4)
d_mean_un = Conv1D(filters=conv[2], kernel_size = K_SIZE,
            padding = PAD_TYPE, strides = 1)(d_up_2) # data_format = "channels_last"
d_log_var_un = Conv1D(filters=conv[2], kernel_size = K_SIZE,
               padding = PAD_TYPE, strides = 1)(d_up_2) #data_format = "channels_last"
d_mean = BatchNormalization(axis = 2)(d_mean_un)
d_log_var = BatchNormalization(axis=2)(d_log_var_un)

# join the branches by generating random normals. this is our reconstruction
x_outputs = Lambda(function = sampling, name='output')([d_mean, d_log_var])

# build classification branch
c_conv_1 = Conv1D(filters = c_conv[0], kernel_size = K_SIZE,
         padding = PAD_TYPE, strides = 1)(latent_inputs)
c_batch_1 = BatchNormalization(axis = 2)(c_conv_1)    # channels along last axis
c_pool_1 = MaxPooling1D(pool_size = RESOLUTION,padding = PAD_TYPE)(c_batch_1)
c_conv_2 = Conv1D(filters = c_conv[1], kernel_size = K_SIZE,
         padding = PAD_TYPE, strides = 1)(c_pool_1)
c_batch_2 = BatchNormalization(axis = 2)(c_conv_2)    # channels along last axis
c_pool_2 = MaxPooling1D(pool_size = RESOLUTION,padding = PAD_TYPE)(c_batch_2)
c_flat = Flatten()(c_pool_2)
c_dense_1 = Dense(units = c_dense[0], activation = ACT_TYPE)(c_flat)
c_batch_1 = BatchNormalization(axis = 1)(c_dense_1)
c_dense_2 = Dense(units = c_dense[1], activation = ACT_TYPE)(c_batch_1)
c_batch_2 = BatchNormalization(axis = 1)(c_dense_2)
y_outputs = Dense(units = NUM_CLASSES, activation = ACT_TYPE)(c_batch_2)

# instantiate subsidiary models
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
decoder = Model(latent_inputs, [d_mean,d_log_var,x_outputs], name='decoder')
classifier = Model(latent_inputs, y_outputs, name='classifier')

# instantiate vae
x_outputs = decoder(encoder(inputs)[2])[2]
x_activated = Activation('linear',name='generation_output')(x_outputs) # need to name output layers
y_outputs = classifier(encoder(inputs)[2])
y_activated = Activation('softmax',name='classification_output')(y_outputs)
vae = Model(inputs, [x_activated,y_activated], name='vae')

# losses
losses = {'generation_output': vae_loss(z_mean,z_log_var,inputs,x_activated,input_shape),
	      'classification_output': LOSS}
loss_weights = {'generation_output': GEN_WEIGHT,
                'classification_output': 1.0}
opt = sgd(lr=0.01,clipnorm=1.)
vae.compile(optimizer=opt, #'adam'
            loss=losses,
            loss_weights=loss_weights)




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
    w=class_weight.compute_class_weight('balanced',np.unique(y_tr),y_tr)
    for n in np.arange(NUM_TIMES):
        #vae.fit(X_tr,y_tr,epochs=ep_at_a_time,batch_size=b_s,validation_data=(X_val,y_val))
        vae.fit(X_tr,{'generation_output':y_tr,'classification_output':y_tr},
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight={'generation_output':w,'classification_output':w},
            validation_data=(X_val,{'generation_output':y_val,'classification_output':y_val}))

        # save visual progress
        window_pred=vae.predict(window[0].reshape(1,x,y))
        window=np.concatenate((window,window_pred),axis=0)

        # save loss
        history=np.stack([vae.history.history['loss'],vae.history.history['val_loss']
                            #,vae.history.history['acc'],vae.history.history['val_acc']]
                            ).T
        loss_history = (history if n==0 else np.concatenate((loss_history,history),axis=0))
    # get the images
    #viz()

    # save weights, loss, windows
    vae.save_weights(os.path.join(dir,'vae.h5'))
    encoder.save_weights(os.path.join(dir,'encoder.h5'))
    decoder.save_weights(os.path.join(dir,'decoder.h5'))
    np.save(os.path.join(dir,'window.npy'),window)
    loss_df = pd.DataFrame()
    loss_df['loss'],loss_df['val_loss']=loss_history[:,0],loss_history[:,1]
    #loss_df['acc'],loss_df['val_acc']=loss_history[:,2],loss_history[:,3]
    loss_df.to_csv(os.path.join(dir,'loss_results.csv'), index=False)
