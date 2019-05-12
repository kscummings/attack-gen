"""
train CNN to detect attacks
agnostic to data version (real or manually/generatively implanted attacks)
"""
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

from keras.layers import Dense, BatchNormalization, Activation, Dropout, Conv1D, Flatten, MaxPooling1D
from keras.models import Model, Sequential
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.utils import plot_model, to_categorical
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import sgd
from keras.callbacks import EarlyStopping

from itertools import chain

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

import format_data
import synthetic_data

'''
CONSTANTS
'''

TEST_SIZE=0.3
BATCH_SIZE=256
MODELS_DIR="best_models"
OUTPUT_DIR="final_results"
NUM_EPOCHS=80
NUM_TRIALS=3

'''
MAIN MODEL
'''

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
    main classifier
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
        opt = sgd(lr = learning_rate,
                  clipnorm = 1. )
        model.compile(loss = 'sparse_categorical_crossentropy',
                      optimizer = opt,
                      metrics = ['accuracy'])

    return model


'''
function to train model once given data
function to test model on official test data
loop function to get average results for one data generation model but multiple draws from the model, record results
'''

def train_attack_gen(data,
                     output_dir,
                     epochs=100,
                     test_size=0.3,
                     batch_size=256,
                     w=np.array([1,1])):
    '''
    train one model
    '''
    os.makedirs(output_dir,exist_ok=True)
    (X,y)=data
    X_tr,X_val,y_tr,y_val=train_test_split(X,y,stratify=y,test_size=test_size,shuffle=True)

    model=build_attack_detection_model()
    model.fit(X_tr,y_tr,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_val,y_val),
              class_weight=w)

    # save loss
    d=model.history.history
    history=np.stack([d['loss'],d['val_loss'],d['acc'],d['val_acc']]).T
    loss_df = pd.DataFrame(history,columns=['loss','val_loss','acc','val_acc'])
    loss_df.to_csv(os.path.join(output_dir,'loss_results.csv'), index=False)

    # save model
    model.save_weights(os.path.join(output_dir,'model.h5'))

    # save data
    data_dir=os.path.join(output_dir, "data")
    os.makedirs(data_dir,exist_ok=True)
    np.save(os.path.join(data_dir,"train"),X_tr)
    np.save(os.path.join(data_dir,"test"),X_val)
    np.save(os.path.join(data_dir,"train_response"),y_tr)
    np.save(os.path.join(data_dir,"test_response"),y_val)

def get_pred(prediction):
    '''
    convert output to binary prediction
    '''
    return (prediction[:,1]>0.5)+0

def train_trials(num_trials,
                 output_dir,
                 models_dir,
                 epochs=100,
                 test_size=0.3,
                 batch_size=256):
    '''
    train attack detection model multiple times, for multiple data generation models
    predict test data with each and record results in global csv
    inputs
    #   num_trials - number of models to train, per data generation model
    #   output_dir - directory to store the results
    #   models_dir - directory where data generation models are stored
    #   epochs - number of epochs per training session
    #   test_size - proportion of training data to use for model validation
    '''
    os.makedirs(output_dir,exist_ok=True)
    r=pd.DataFrame(['model']*4*num_trials,columns=['model_type'])
    results=pd.DataFrame(np.zeros((4*num_trials,9)),
        columns=['loss','val_loss','acc','val_acc','test_acc','test_tn','test_fp','test_fn','test_tp'])
    results=pd.concat([r,results],axis=1)
    results.to_csv(os.path.join(output_dir,"final_results.csv"), index=False)

    # get data
    (clean,y_clean),(attack,y_attack),names=format_data.get_rolled_data()
    test,y_test=format_data.get_rolled_test_data()

    # get weights
    decoder_weights=os.path.join(models_dir,"decoder.h5")
    classifier_weights=os.path.join(models_dir,"classifier.h5")
    clean_decoder_weights=os.path.join(models_dir,"clean_decoder.h5")

    w=class_weight.compute_class_weight('balanced',np.unique(y_attack),y_attack)

    # TODO: clean this up later
    for n in np.arange(num_trials):

        ########################## REAL DATA ONLY
        current_dir=os.path.join(output_dir,"baseline_{}".format(n+1))
        train_attack_gen(data=(attack,y_attack),
                         output_dir=current_dir,
                         epochs=epochs,
                         test_size=test_size,
                         batch_size=batch_size,
                         w=w)
        res=pd.read_csv(os.path.join(current_dir,"loss_results.csv"))
        test_model=build_attack_detection_model()
        test_model.load_weights(os.path.join(current_dir,"model.h5"))
        tn,fp,fn,tp = confusion_matrix(y_test,get_pred(test_model.predict(test))).ravel()

        results.loc[4*n,'model_type']='baseline'
        results.loc[4*n,1:5]=np.array(res.iloc[len(res)-1])
        results.loc[4*n,'test_acc']=(tn+tp)/len(y_test)
        results.loc[4*n,6:]=np.array([tn,fp,fn,tp])
        results.to_csv(os.path.join(output_dir,"final_results.csv"), index=False)

        ########################## WITH SYNTHETIC SWAPS
        current_dir=os.path.join(output_dir,"synth_swaps_{}".format(n+1))
        data=synthetic_data.get_synthetic_training_data(decoder_weights,
                                         classifier_weights,
                                         clean_decoder_weights,
                                         implant_synth_attacks=False,
                                         implant_manual_attacks=True)
        train_attack_gen(data=data,
                         output_dir=current_dir,
                         epochs=epochs,
                         test_size=test_size,
                         batch_size=batch_size)
        res=pd.read_csv(os.path.join(current_dir,"loss_results.csv"))
        test_model=build_attack_detection_model()
        test_model.load_weights(os.path.join(current_dir,"model.h5"))
        tn,fp,fn,tp = confusion_matrix(y_test,get_pred(test_model.predict(test))).ravel()

        results.loc[4*n+1,'model_type']='synth_swaps'
        results.loc[4*n+1,1:5]=np.array(res.iloc[len(res)-1])
        results.loc[4*n+1,'test_acc']=(tn+tp)/len(y_test)
        results.loc[4*n+1,6:]=np.array([tn,fp,fn,tp])
        results.to_csv(os.path.join(output_dir,"final_results.csv"), index=False)

        ########################## WITH SYNTHETIC GENERATED
        current_dir=os.path.join(output_dir,"synth_gen_{}".format(n+1))
        data=synthetic_data.get_synthetic_training_data(decoder_weights,
                                         classifier_weights,
                                         clean_decoder_weights,
                                         implant_synth_attacks=True,
                                         implant_manual_attacks=False)
        train_attack_gen(data=data,
                         output_dir=current_dir,
                         epochs=epochs,
                         test_size=test_size,
                         batch_size=batch_size)
        res=pd.read_csv(os.path.join(current_dir,"loss_results.csv"))
        test_model=build_attack_detection_model()
        test_model.load_weights(os.path.join(current_dir,"model.h5"))
        tn,fp,fn,tp = confusion_matrix(y_test,get_pred(test_model.predict(test))).ravel()

        results.loc[4*n+2,'model_type']='synth_swaps'
        results.loc[4*n+2,1:5]=np.array(res.iloc[len(res)-1])
        results.loc[4*n+2,'test_acc']=(tn+tp)/len(y_test)
        results.loc[4*n+2,6:]=np.array([tn,fp,fn,tp])
        results.to_csv(os.path.join(output_dir,"final_results.csv"), index=False)

        ########################## BOTH SWAPS AND GENERATED
        current_dir=os.path.join(output_dir,"both_{}".format(n+1))
        data=synthetic_data.get_synthetic_training_data(decoder_weights,
                                         classifier_weights,
                                         clean_decoder_weights,
                                         implant_synth_attacks=True,
                                         implant_manual_attacks=True)
        train_attack_gen(data=data,
                         output_dir=current_dir,
                         epochs=epochs,
                         test_size=test_size,
                         batch_size=batch_size)
        res=pd.read_csv(os.path.join(current_dir,"loss_results.csv"))
        test_model=build_attack_detection_model()
        test_model.load_weights(os.path.join(current_dir,"model.h5"))
        tn,fp,fn,tp = confusion_matrix(y_test,get_pred(test_model.predict(test))).ravel()

        results.loc[4*n+3,'model_type']='both'
        results.loc[4*n+3,1:5]=np.array(res.iloc[len(res)-1])
        results.loc[4*n+3,'test_acc']=(tn+tp)/len(y_test)
        results.loc[4*n+3,6:]=np.array([tn,fp,fn,tp])
        results.to_csv(os.path.join(output_dir,"final_results.csv"), index=False)

'''
MAIN LOOP
'''

def main():
    train_trials(num_trials=NUM_TRIALS,
                 output_dir=output_dir,
                 models_dir=models_dir,
                 epochs=NUM_EPOCHS,
                 test_size=TEST_SIZE,
                 batch_size=BATCH_SIZE)

if __name__ == '__main__':
    main()
