from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


########## preprocess raw data

def roll(x, y, window_length):
    """
    roll data into windows
    ### Keywords
    *`x` - covariates
    *`y` - labels
    """
    n,m=x.shape[0]-window_length+1,x.shape[1] # obs,feat
    x_rolled=np.zeros((n,window_length,m))
    y_rolled=np.zeros(n)
    for w in range(n):
        x_rolled[w]=x[w:(w+window_length)]
        y_rolled[w]=np.any(y[w:(w+window_length)]==1).astype(int)

    return (x_rolled,y_rolled)


def get_rolled_data(data_path,std_pct=0.1,window_length=24):
    """
    import, standardize, remove low-variance features, and roll
    ### Keywords
    *`data_path` - directory containing data
    ### Returns
    """
    # import
    clean=pd.read_csv(os.path.join(data_path,"BATADAL_dataset03.csv"))
    attack=pd.read_csv(os.path.join(data_path,"BATADAL_dataset04_manualflags.csv"))
    test=pd.read_csv(os.path.join(data_path,"BATADAL_test_dataset_withflags.csv"))

    # feature/response format
    y_clean=clean['ATT_FLAG'].values
    y_attack=attack['ATT_FLAG'].values
    y_test=test['ATT_FLAG'].values
    clean=clean.drop(columns=['DATETIME','ATT_FLAG'])
    attack=attack.drop(columns=['DATETIME','ATT_FLAG']).values
    test=test.drop(columns=['DATETIME','ATT_FLAG'])

    names=clean.columns
    clean=clean.values            # keras takes np arrays
    test=test.values

    # standardize data and remove low-variance features
    scaler=MinMaxScaler()
    scaler.fit(clean)
    clean=scaler.fit_transform(clean)
    attack=scaler.transform(attack)
    test=scaler.transform(test)
    keep=clean.std(0) > np.percentile(clean.std(0), std_pct)
    clean=clean[:,keep]
    attack=attack[:,keep]
    test=test[:,keep]
    names=names[keep]

    # roll data into windows
    clean_roll,y_clean_roll=roll(clean,y_clean,window_length)
    attack_roll,y_attack_roll=roll(attack,y_attack,window_length)
    test_roll,y_test_roll=roll(test,y_test,window_length)

    return (clean_roll, y_clean_roll), (attack_roll, y_attack_roll), (test_roll, y_test_roll), names

def get_rolled_attack_data(data_path,std_pct=0.1,window_length=24):
    """
    Get training examples to train discriminator
    """
    (_, _), (attack, y_attack), (_,_) _ = get_rolled_data(data_path,std_pct,window_length)
    return attack[y_attack==1,:]



# augment clean data with attacks, given strategy or using default strategy
# given strategy: manipulating sensor readings according to induced distribution over sensor sets
# default strategy: manipulating sensor readings uniformly


# input: sensor set selections
# how to collapse distribution?


######### function to generate sensor distributions from result set





######## function to perform a swap



######## function to perform all swaps according to a distribution
