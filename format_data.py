"""
script to treat data
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

import VAE_attack_generator

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
    clean = pd.read_csv("./batadal/BATADAL_dataset03.csv")
    attack = pd.read_csv("./batadal/BATADAL_dataset04_manualflags.csv")

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

'''
GENERATE SYNTHETIC DATA
'''

def gen_data(decoder_weights,
             classifier_weights,
             num_obs,
             latent_input_shape=(6,100),
             include_attacks=True,
             attack_prop=0.5):
    '''
    load in decoder (and classifier), then generate new data
    inputs
    #   decoder weights - path of h5 file with decoder weights
    #   classifier_weights - path of h5 file with classifier weights
    #   num_obs - size of data to generate
    #   include_attacks - whether i want to generate attacks too
    #   attack_prop - proportion of observations that are attacks
    '''
    noise=np.random.normal(size=(num_obs,latent_input_shape[0],latent_input_shape[1]))
    # vae=VAE_attack_generator.get_vae()
    # vae.load_weights(weights)
    decoder=VAE_attack_generator.build_decoder()
    decoder.load_weights(decoder_weights)
    synthetic_features=decoder.predict(noise)[2]

    if include_attacks:
        classifier=VAE_attack_generator.build_classifier()
        classifier.load_weights(classifier_weights)
        synthetic_labels=(classifier.predict(noise)[:,1]>=0.5)+0

        # get right proportion of attacks
        num_left=int(np.ceil(num_obs*attack_prop)-sum(synthetic_labels))
        while num_left > 0:
            noise=np.random.normal(size=(num_obs,latent_input_shape[0],latent_input_shape[1]))

            # new observations to choose from, only keep new attacks
            new_synth_lab=(classifier.predict(noise)[:,1]>=0.5)+0
            new_synth_feat=decoder.predict(noise)[2][np.where(new_synth_lab==1)]

            # candidate replacement locations
            clean_locations=np.where(synthetic_labels==0)[0]

            # either generated more than necessary or not enough
            n = (num_left if len(new_synth_feat) >= num_left else len(new_synth_feat))

            replace=np.random.choice(clean_locations,n,replace=False)
            synthetic_features[replace]=new_synth_feat
            synthetic_labels[replace]=np.ones(n)

            num_left -= n

    return (synthetic_features, synthetic_labels)


# function to manually implant attacks, given model-agnostic input data
def gen_data_manual(features,
                    names):
    '''
    given clean data, manually transform into naive attacks (swaps)
    transform all data given
    inputs
    #   features - data to transform
    #   names - feature labels
    '''
    m=len(features)
    window_length=features.shape[1]

    # get sensor type for each feature
    # build dictionary of sensor type => feature indices
    # note: prefixes of feature names give measurement types (e.g. pressure vs. flow)
    tags=['L','F','S','P']
    keys=np.array([str.split(names[i],'_')[0] for i in range(len(names))])
    sensor_dict={type:np.where(keys==type)[0] for type in tags}

    # get features to swap in each observation
    sensor_type=np.random.choice(tags,size=m,replace=True)
    swap_pair=np.stack([np.random.choice(sensor_dict[type],size=2,replace=False) for type in sensor_type])

    # get random ranges to swap
    att_len=np.random.choice(np.arange(1,window_length+1),size=m,replace=True)
    att_start=list(np.random.choice(range(window_length-att_len[k]+1),size=1)[0] for k in range(m))
    att_end=att_start+att_len
    # perform the swaps (not vectorized...)
    for k in np.arange(m):
        start=att_start[k]
        end=att_end[k]

        # check if swap is "good enough",
        # i.e. the difference between the swapped readings is non-negligible (greater than 0.1)
        swap_quality=np.any(np.round(features[k,start:end,swap_pair[k,0]]-
            features[k,start:end,swap_pair[k,1]],1)!=0)
        while !swap_quality:
            sensor_type[k]=np.random.choice(tags,size=1,replace=True)
            swap_pair[k]=np.random.choice(sensor_dict[sensor_type[k]],size=2,replace=False)
            swap_quality=np.any(np.round(features[k,start:end,swap_pair[k,0]]-
                features[k,start:end,swap_pair[k,1]],1)!=0)

        # do the swap
        temp=features[k,start:end,swap_pair[k,0]].copy()
        features[k,start:end,swap_pair[k,0]]=features[k,start:end,swap_pair[k,1]]
        features[k,start:end,swap_pair[k,1]]=temp
        temp=None

    return features


# wrapper to manually implant attacks in synthetic "clean" data
