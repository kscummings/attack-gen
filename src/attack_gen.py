from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

TAGS=['L','F','S','P']
SENSOR_SETS={
    "s1":1,
    "s2":2,
    "s3":3,
    "s4":4,
    "s5":5
 }


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
    don't swap switches
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
    sensor_set={i:[name for name in ss if not name.startswith("S")] for (i,ss) in sensor_set.items()}
    sensor_set={i:[np.where(names==name)[0][0] for name in ss] for (i,ss) in sensor_set.items()}
    return sensor_set

def read_sensordist(res_path):
    sdist=pd.read_csv(path.join(res_path,"sdist.csv"))
    dists={}
    for i in range(len(sdist)):
        trial_type=sdist.at[i,"trial_type"]
        budget=sdist.at[i,"budget"]
        sdist_id=sdist.at[i,"sdist_id"]
        if (trial_type!="full") & (budget!=0):
            sd={ss_id:sdist.at[i,ss_var] for ss_var,ss_id in SENSOR_SETS.items()}
            sd={ss:total/sum(sd.values()) for (ss,total) in sd.items()}
            dists[sdist_id]=sd
    return dists

######## generate synthetic data

def manual_attacks(x,names,sdist,sensorset_dict):
    '''
    given clean data, manually transform into naive attacks (swaps)
    transform all data given
    ### Keywords
    #`x`-data to transform
    #`names` - feature labels
    *`sdist` - distribution over sensor sets
    *`sensorset_dict` - (id => list of sensor variable names)
    '''
    features=x.copy()
    m=len(features)
    window_length=features.shape[1]

    # number of samples from each sensor set type
    sdist={i:int(np.floor(p*m)) for (i,p) in sdist.items()}
    sensorsets=list(sdist.keys())

    # get observations to swap for each
    num_ind=sum(sdist.values())
    indices=[i for i in range(num_ind)] # already shuffled outside loop
    breaks=np.cumsum(list(sdist.values()))
    breaks=[0]+list(breaks[:-1])+[num_ind]
    swap_ind={i:indices[breaks[(i-1)]:breaks[i]] for i in range(1,6)}

    # perform swaps from each sensor set
    for ss in sensorsets:
        if swap_ind[ss]:
            # get features to swap in each observation
            swap_pair=np.stack([np.random.choice(sensorset_dict[ss],size=2,replace=False) for _ in range(sdist[ss])])

            # get random ranges to swap
            att_len=np.random.choice(np.arange(1,window_length+1),size=sdist[ss],replace=True)
            att_start=list(np.random.choice(range(window_length-att_len[k]+1),size=1)[0] for k in range(sdist[ss]))
            att_end=att_start+att_len

            # perform the swaps on a subset of the data
            for k in range(len(swap_ind[ss])):
                start=att_start[k]
                end=att_end[k]
                k_ind=swap_ind[ss][k]

                # check if swap is "good enough",
                # i.e. the difference between the swapped readings is non-negligible (greater than 0.1)
                swap_quality=np.any(np.round(features[k_ind,start:end,swap_pair[k,0]]-features[k_ind,start:end,swap_pair[k,1]],1)!=0)
                while not swap_quality:
                    swap_pair[k]=np.random.choice(sensorset_dict[ss],size=2,replace=False)
                    swap_quality=np.any(np.round(features[k_ind,start:end,swap_pair[k,0]]-features[k_ind,start:end,swap_pair[k,1]],1)!=0)

                # do the swap
                temp0,temp1=copy(features[k_ind,start:end,swap_pair[k,0]]),copy(features[k_ind,start:end,swap_pair[k,1]])
                features[k_ind,start:end,swap_pair[k,0]],features[k_ind,start:end,swap_pair[k,1]]=temp1,temp0


    # not all features got swapped bc of rounding
    return features[:num_ind]

def data_generation(data_path,trials_dir,num_gen):

    # world
    sensordist_path=path.join(data_path,trials_dir)
    dists=read_sensordist(sensordist_path) # sample distributions
    (clean,y_clean),_,_,names=get_rolled_data(data_path) # raw data
    sensorset_dict=get_sensor_sets(data_path,names) # sensor sets

    # for shuffling later
    indices=[i for i in range(len(clean))]
    halfway_ind=int(np.floor(len(indices)/2))

    # write output dir
    output_dir=path.join(data_path,trials_dir,"sim_data")
    try:
        os.makedirs(output_dir)
    except OSError as e:
        raise e

    # simulate attacks
    for sdist_id,sdist in dists.items():
        for i in range(num_gen):
            # randomly sample half
            np.random.shuffle(indices)
            attack=manual_attacks(clean[indices[:halfway_ind]],names,sdist,sensorset_dict)
            num_attacks=attack.shape[0]
            sim_feat=np.vstack((attack,clean[indices[num_attacks:]]))
            sim_labels=np.hstack((np.ones(num_attacks),np.zeros(len(indices)-num_attacks)))

            # preserve
            data=(sim_feat,sim_labels)
            pickle_fp=path.join(output_dir,"sim_data_%s_%d.pickle"%(sdist_id,i+1))
            with open(pickle_fp, "wb") as f:
                pickle.dump(data, f)
