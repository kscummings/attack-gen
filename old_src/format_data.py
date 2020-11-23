"""
script to treat data:
import, roll, generate, and manually change
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from sklearn.preprocessing import MinMaxScaler

import os
import pandas as pd
import numpy as np

'''
HELPER FUNCTIONS
import unaltered data
'''

def roll(x, y, window_length):
    """
    roll data into windows
    """
    n = x.shape[0] - window_length + 1 # new num obs
    m = x.shape[1] # num features
    x_rolled = np.zeros((n,window_length,m))
    y_rolled = np.zeros(n)
    for w in range(n):
        x_rolled[w] = x[w:(w+window_length)]
        y_rolled[w] = np.any(y[w:(w+window_length)]==1).astype(int)

    return (x_rolled, y_rolled)


def get_rolled_data(std_pct=0.1,window_length=24):
    """
    import, standardize, remove low-variance features, and roll
    """
    # import
    clean = pd.read_csv(os.path.join("batadal","BATADAL_dataset03.csv"))
    attack = pd.read_csv(os.path.join("batadal","BATADAL_dataset04_manualflags.csv"))

    # feature/response format
    y_clean = clean['ATT_FLAG'].values
    y_attack = attack['ATT_FLAG'].values
    clean = clean.drop(columns=['DATETIME','ATT_FLAG'])
    attack = attack.drop(columns=['DATETIME','ATT_FLAG']).values

    names = clean.columns
    clean = clean.values            # keras takes np arrays

    # standardize data and remove low-variance features
    scaler = MinMaxScaler()
    scaler.fit(clean)
    clean = scaler.fit_transform(clean)
    attack = scaler.transform(attack)
    keep = clean.std(0) > np.percentile(clean.std(0), std_pct)
    clean = clean[:,keep]
    attack = attack[:,keep]
    names = names[keep]

    # roll data into windows
    clean_roll, y_clean_roll = roll(clean, y_clean, window_length)
    attack_roll, y_attack_roll = roll(attack, y_attack, window_length)

    return (clean_roll, y_clean_roll), (attack_roll, y_attack_roll), names

def get_rolled_attack_data(std_pct=0.1,window_length=24):
    """
    Get training examples to train discriminator
    """
    (_, _), (attack, y_attack), _ = get_rolled_data(std_pct,window_length)
    return attack[y_attack==1,:]

def get_rolled_test_data(std_pct=0.1,window_length=24):
    '''
    import/process official test set
    '''
    clean=pd.read_csv(os.path.join("batadal","BATADAL_dataset03.csv"))
    test=pd.read_csv(os.path.join("batadal","BATADAL_test_dataset_withflags.csv"))
    y_test=test['ATT_FLAG'].values

    clean=clean.drop(columns=['DATETIME','ATT_FLAG']).values
    test=test.drop(columns=['DATETIME','ATT_FLAG'])
    names=test.columns
    test=test.values

    scaler = MinMaxScaler()
    scaler.fit(clean)
    test = scaler.transform(test)
    keep = clean.std(0) > np.percentile(clean.std(0), std_pct)
    test = test[:,keep]
    names = names[keep]

    return roll(test, y_test, window_length)
