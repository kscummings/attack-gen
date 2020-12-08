from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from glob import glob
from itertools import chain

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

from scipy.stats import multivariate_normal

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from attack_gen import get_rolled_data


'''
CONSTANTS
'''

DATA_PATH="/Users/kaylacummings/Dropbox (MIT)/batadal" # get_data_path()
TRIAL_DIR="interdiction_results"
OUTPUT_DIR="train_results"
SYNTHDATA_DIR="sim_data"

TEST_SIZE=0.3
BATCH_SIZE=256
NUM_EPOCHS=80
NUM_TRIALS=3

# front matter ..
TRIAL_PATH=path.join(DATA_PATH,TRIAL_DIR)
OUTPUT_PATH=path.join(TRIAL_PATH,OUTPUT_DIR)
SYNTHDATA_PATH=path.join(TRIAL_PATH,SYNTHDATA_DIR)
tf.compat.v1.disable_eager_execution()



def build_attack_detection_model(conv_layers = [50,50,50],
                                 dense_layers = [10,10],
                                 k_size = 5,
                                 pad_type = 'same',
                                 resolution = 2,
                                 kernel_initializer = 'glorot_uniform',
                                 num_classes = 2,
                                 input_shape = (24,36),
                                 learning_rate = .01,
                                 reg = .00,
                                 compile = True):
    """
    CNN classifier
    """
    model = Sequential()
    model.add(Conv1D(input_shape=input_shape,
                  filters = conv_layers[0],
                  kernel_size = k_size,
                  padding = pad_type,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer = l2(reg),
                  bias_initializer = 'ones',
                  bias_regularizer = l2(reg)))
    model.add(MaxPooling1D(pool_size = resolution, padding = pad_type))
    model.add(BatchNormalization(axis=2))

    n_conv = len(conv_layers)
    n_dense = len(dense_layers)

    for i in range(1, n_conv):
        model.add(Conv1D(filters = conv_layers[i],
                  kernel_size = k_size,
                  padding = pad_type,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer = l2(reg),
                  bias_initializer = 'ones',
                  bias_regularizer = l2(reg)))
        model.add(MaxPooling1D(pool_size = resolution, padding = pad_type))
        model.add(BatchNormalization(axis=2))
    model.add(Flatten())
    for i in range(n_dense):
        model.add(Dense(dense_layers[i],
            kernel_initializer=kernel_initializer,
            kernel_regularizer= l2(reg),
            bias_initializer = 'ones',
            bias_regularizer = l2(reg),
            activation = 'relu'))
        model.add(BatchNormalization())

    if compile:
        model.add(Dense(num_classes, activation = 'softmax'))
        opt = SGD(lr = learning_rate,
                  clipnorm = 1. )
        model.compile(loss = 'sparse_categorical_crossentropy',
                      optimizer = opt,
                      metrics = ['accuracy'])

    return model



def train_attack_gen(data,
                     output_dir,
                     model_name,
                     epochs=100,
                     test_size=0.3,
                     batch_size=256,
                     w=np.array([1,1])):
    """
    Train CNN
    """
    try:
        os.makedirs(output_dir,exist_ok=True)
    except OSError as e:
        raise e

    (X,y),(X_test,y_test)=data
    X_tr,X_val,y_tr,y_val=train_test_split(X,y,stratify=y,test_size=test_size,shuffle=True)

    model=build_attack_detection_model()
    model.fit(X_tr,y_tr,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_val,y_val),
              class_weight=w)

    # save loss
    d=model.history.history
    history=np.stack([d['loss'],d['val_loss'],d['accuracy'],d['val_accuracy']]).T
    loss_df=pd.DataFrame(history,columns=['loss','val_loss','acc','val_acc'])
    loss_dir=os.path.join(output_dir, "loss")
    if not path.isdir(loss_dir):
        os.makedirs(loss_dir)
    loss_df.to_csv(os.path.join(loss_dir,'loss_%s.csv'%(model_name)), index=False)

    # save model
    model_dir=os.path.join(output_dir, "models")
    if not path.isdir(model_dir):
        os.makedirs(model_dir)
    model.save_weights(os.path.join(model_dir,'model_%s.h5'%(model_name)))

    # save data
    data_dir=os.path.join(output_dir, "train_data")
    if not path.isdir(data_dir):
        os.makedirs(data_dir)
    np.save(os.path.join(data_dir,"train_%s"%(model_name)),X_tr)
    np.save(os.path.join(data_dir,"test_%s"%(model_name)),X_val)
    np.save(os.path.join(data_dir,"train_label_%s"%(model_name)),y_tr)
    np.save(os.path.join(data_dir,"test_label_%s"%(model_name)),y_val)

    # return test results
    return  confusion_matrix(y_test,get_pred(model.predict(X_test))).ravel()

def get_pred(prediction):
    '''
    convert output to binary prediction
    '''
    return (prediction[:,1]>0.5)+0

def train_trials(num_trials,
                 output_dir,
                 synth_data_dir,
                 batadal_data_dir,
                 epochs=100,
                 test_size=0.3,
                 batch_size=256):
    '''
    train attack detection model multiple times, for multiple data generation models
    predict test data with each and record results in global csv
    inputs
    *`num_trials`-number of models to train per dataset
    *`output_dir`-directory to store the results
    *`synth_data_dir`-directory where synthetic training datasets are stored
    *`batadal_data_dir`-directory with original training and test sets
    *`epochs`-number of epochs per training session
    *`test_size`-proportion of training data to use for model validation
    '''
    # don't overwrite
    try:
        os.makedirs(output_dir,exist_ok=False)
    except OSError as e:
        raise e

    # synthetic data locations
    synth_filenames=glob(path.join(synth_data_dir,"*.pickle"))
    num_synth=len(synth_filenames)

    # results location
    cols=['model_name','trial','loss','val_loss','acc','val_acc','test_acc','test_tn','test_fp','test_fn','test_tp']
    results=pd.DataFrame(columns=cols)
    finalres_filename=os.path.join(output_dir,"final_results.csv")
    results.to_csv(finalres_filename,index=False)

    # test data
    _,_,(test,y_test),_=get_rolled_data(batadal_data_dir)

    record_ind=0
    for fn in synth_filenames[:3]:
        # get data
        with open(fn,"rb") as f:
            X,y=pickle.load(f)
        root_model_name=fn.replace(synth_data_dir,"").replace("/sim_data_","").replace(".pickle","")

        for n in range(num_trials):

            model_name="%s_%d"%(root_model_name,n)
            tn,fp,fn,tp=train_attack_gen(
                             data=((X,y),(test,y_test)),
                             output_dir=output_dir,
                             model_name=model_name,
                             epochs=epochs,
                             test_size=test_size,
                             batch_size=batch_size,
                             w=w)


            # populate final results df
            res=pd.read_csv(os.path.join(output_dir,"loss","loss_%s.csv"%(model_name)))
            row=np.hstack(([root_model_name,n],np.array(res.iloc[len(res)-1]),[(tn+tp)/len(y_test)],[tn,fp,fn,tp]))
            results=pd.DataFrame(row.reshape(1,row.shape[0]),columns=cols)
            results.to_csv(finalres_filename,index=False,mode='a',header=False)


'''
MAIN LOOP
'''

def main():

    train_trials(num_trials=NUM_TRIALS,
                 output_dir=OUTPUT_PATH,
                 synth_data_dir=SYNTHDATA_PATH,
                 batadal_data_dir=DATA_PATH,
                 epochs=NUM_EPOCHS,
                 test_size=0.3,
                 batch_size=256)


if __name__ == '__main__':
    main()
