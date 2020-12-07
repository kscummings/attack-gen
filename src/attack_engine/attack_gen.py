from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

TAGS=['L','F','S','P']
SENSOR_SETS=["s1","s2","s3","s4","s5"]


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
    (_,_), (attack, y_attack), (_,_), _ = get_rolled_data(data_path,std_pct,window_length)
    return attack[y_attack==1,:]

def get_sensor_sets(data_path,names):
    """
    ### Keywords
    *`data_path` - data location
    *`names` - batadal data columns
    ### Returns
    *dict from sensor set id to variable index
    """
    filepath=os.path.join(data_path,"sensor_xwalk.csv")
    dat=pd.read_csv(filepath)
    sensors=dat.columns[dat.columns!='sensor_set_id']
    sensor_set={dat.at[i,"sensor_set_id"]:[s for s in sensors if dat.at[i,s]==1] for i in range(dat.shape[0])}
    sensor_set={i:[name for name in names if any([sensor in name for sensor in sensor_set])] for (i,sensor_set) in sensor_set.items()}
    sensor_set={i:[np.where(names==name)[0][0] for name in ss] for (i,ss) in sensor_set.items()}
    return sensor_set

def read_sensordist(res_path):
    sdist=pd.read_csv(path.join(res_path,"sdist.csv"))
    dists={}
    for i in range(len(sdist)):
        trial_type=sdist.at[i,"trial_type"]
        trial_param=sdist.at[i,"trial_param"]
        fortify=sdist.at[i,"fortify"]
        budget=sdist.at[i,"budget"]
        if trial_type!="full":
            sd={ss:sdist.at[i,ss] for ss in SENSOR_SETS}
            sd={ss:total/sum(sd.values()) for (ss,total) in sd.items()}
            dists[(trial_type,trial_param,budget,fortify)]=sd
    return dists

######## function to perform a swap

def manual_attacks(x,names,sdist,sensorset_dict):
    '''
    given clean data, manually transform into naive attacks (swaps)
    transform all data given
    ### Keywords
    #`x`-data to transform
    #`names` - feature labels
    *`sdist` - distribution over sensor sets
    *`sensor_dict` - (id => list of sensor variable names)
    '''
    features=x.copy()
    m=len(features)
    window_length=features.shape[1]

    # number of samples from each sensor set type
    sdist={i:int(np.floor(p*m)) for (i,p) in sdist.items()}
    sensorsets=list(sdist.keys())

    for ss in sensorsets:
        # perform sdist[ss] swaps
        # get features to swap in each observation
        swap_pair=np.stack([np.random.choice(sensorset_dict[ss],size=2,replace=False) for _ in range(sdist[ss])])

        # get random ranges to swap
        att_len=np.random.choice(np.arange(1,window_length+1),size=sdist[ss],replace=True)
        att_start=list(np.random.choice(range(window_length-att_len[k]+1),size=1)[0] for k in range(sdist[ss]))
        att_end=att_start+att_len

        # perform the swaps
        for k in np.arange(m):
            start=att_start[k]
            end=att_end[k]

            # check if swap is "good enough",
            # i.e. the difference between the swapped readings is non-negligible (greater than 0.1)
            swap_quality=np.any(np.round(features[k,start:end,swap_pair[k,0]]-features[k,start:end,swap_pair[k,1]],1)!=0)
            while not swap_quality:
                sensor_type[k]=np.random.choice(TAGS,size=1)[0]
                swap_pair[k]=np.random.choice(sensor_dict[sensor_type[k]],size=2,replace=False)
                swap_quality=np.any(np.round(features[k,start:end,swap_pair[k,0]]-features[k,start:end,swap_pair[k,1]],1)!=0)

            # do the swap
            features[k,start:end,swap_pair[k,0]],features[k,start:end,swap_pair[k,1]]=features[k,start:end,swap_pair[k,1]],features[k,start:end,swap_pair[k,0]]

    return features
