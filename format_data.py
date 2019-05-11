"""
script to treat data:
import, roll, generate, and manually change
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

def synth_attacks(decoder_weights,
                  classifier_weights,
                  num_obs,
                  latent_input_shape=(6,100)):
    '''
    load in decoder (and classifier), then generate attacks
    inputs
    #   decoder weights - path of h5 file with decoder weights
    #   classifier_weights - path of h5 file with classifier weights
    #   num_obs - number of attacks to generate
    #   include_attacks - whether i want to generate attacks too
    '''

    noise=np.random.normal(size=(int(num_obs),latent_input_shape[0],latent_input_shape[1]))

    decoder=build_decoder()#VAE_attack_generator.build_decoder()
    decoder.load_weights(decoder_weights)
    synthetic_features=decoder.predict(noise)[2]

    classifier=build_classifier()#VAE_attack_generator.build_classifier()
    classifier.load_weights(classifier_weights)
    synthetic_labels=(classifier.predict(noise)[:,1]>=0.5)+0

    # only want attacks
    num_left=int(num_obs-sum(synthetic_labels))
    while num_left > 0:
        noise=np.random.normal(size=(int(num_obs),latent_input_shape[0],latent_input_shape[1]))

        # new observations to choose from, only keep new attacks
        new_synth_lab=(classifier.predict(noise)[:,1]>=0.5)+0
        new_synth_feat=decoder.predict(noise)[2][np.where(new_synth_lab==1)]

        # candidate replacement locations
        clean_locations=np.where(synthetic_labels==0)[0]

        # either generated more than necessary or not enough
        n = (num_left if len(new_synth_feat) >= num_left else len(new_synth_feat))

        replace=np.random.choice(clean_locations,n,replace=False)
        synthetic_features[replace]=new_synth_feat[:n] # might have generated too much
        synthetic_labels[replace]=np.ones(n)

        num_left -= n

    return synthetic_features


# function to manually implant attacks, given model-agnostic input data
def manual_attacks(x,
                   names):
    '''
    given clean data, manually transform into naive attacks (swaps)
    transform all data given
    inputs
    #   x - data to transform
    #   names - feature labels
    '''
    features=x.copy()
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
        while not swap_quality:
            sensor_type[k]=np.random.choice(tags,size=1)[0]
            swap_pair[k]=np.random.choice(sensor_dict[sensor_type[k]],size=2,replace=False)
            swap_quality=np.any(np.round(features[k,start:end,swap_pair[k,0]]-
                features[k,start:end,swap_pair[k,1]],1)!=0)

        # do the swap
        features[k,start:end,swap_pair[k,0]],features[k,start:end,swap_pair[k,1]]=features[k,start:end,swap_pair[k,1]],features[k,start:end,swap_pair[k,0]]

    return features


def synth(num_obs,
          decoder_weights,
          man_att=True,
          latent_input_shape=(6,100)):
    '''
    generate clean synthetic data, with option to manually convert to attacks
    inputs
    #   num_obs - desired number of attacks
    #   decoder_weights - path to h5 file with decoder weights
    #   latent_input_shape - dimensions of encoded latent space
    return
    #   synthetic attacks
    '''
    (_,_),(_,_),names=get_rolled_data()
    noise=np.random.normal(size=(int(num_obs),latent_input_shape[0],latent_input_shape[1]))

    decoder=build_decoder()#VAE_attack_generator.build_decoder()
    decoder.load_weights(decoder_weights)
    clean=decoder.predict(noise)[2]

    return manual_attacks(clean,names) if man_att else clean

def get_synthetic_training_data(attack_decoder_weights,
                                classifier_weights,
                                clean_decoder_weights,
                                implant_synth_attacks,
                                implant_manual_attacks,
                                latent_input_shape=(6,100)):
    '''
    generate synthetic data with desired proportion and type of attacks
    always generates synthetic data on top of batadal03
    want to quantify added value of synthetic data

    model: always include full batadal04 set. augment with clean data type
    and attack data type to the desired proportions. augmentations come from
    batadal03.

    options:
    (0)     always include real attack examples in training, to quantify
            added benefit of the synthetic data
    (1)     manual attacks on synthetic data or generated attacks or both
    (2)     implant in raw or generated clean data
    inputs
    #       attack_decoder_weights - path to h5 file with attack decoder weights
    #       classifier_weights
    #       clean_decoder_weights
    #       synth_attacks - boolean, generate attacks from decoder?
    #       manual_attacks - boolean, generate manual attacks?
    #       latent_input_shape - dimensions of encoded space
    '''

    (X_clean,y_clean),(X,y),names=get_rolled_data()

    # include every clean observation regardless
    X=np.concatenate((X,X_clean),axis=0)
    y=np.concatenate((y,y_clean))

    # get attack augmentations
    # not the most efficient way to code this but ya girl is rushing to the finish line!
    num_clean=len(X)-sum(y)
    num_attacks_left=np.ceil(num_clean-sum(y)) # want it to be balanced
    if implant_synth_attacks & implant_manual_attacks:
        num_synth_attacks=np.ceil(num_attacks_left/2)
        X_synthetic=synth_attacks(decoder_weights=attack_decoder_weights,
                                  classifier_weights=classifier_weights,
                                  num_obs=num_synth_attacks,
                                  latent_input_shape=latent_input_shape)
        X_manual=synth(num_obs=num_attacks_left-num_synth_attacks,
                       decoder_weights=clean_decoder_weights,
                       man_att=True,
                       latent_input_shape=latent_input_shape)
        X_attack=np.concatenate((X_synthetic,X_manual),axis=0)
    elif implant_synth_attacks:
        X_attack=synth_attacks(decoder_weights=attack_decoder_weights,
                               classifier_weights=classifier_weights,
                               num_obs=num_attacks_left,
                               latent_input_shape=latent_input_shape)
    else: #only manual-on-synth attacks default
        X_attack=synth(num_obs=num_attacks_left,
                       decoder_weights=clean_decoder_weights,
                       man_att=True,
                       latent_input_shape=latent_input_shape)

    # append synthetic attacks to training set
    X=np.concatenate((X,X_attack),axis=0)
    y=np.concatenate((y,np.ones(int(num_attacks_left))))

    return (X,y)
