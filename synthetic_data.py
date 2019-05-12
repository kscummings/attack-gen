'''
generate synthetic data
'''
import numpy as np

import format_data
import VAE_attack_generator

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

    decoder=VAE_attack_generator.build_decoder()
    decoder.load_weights(decoder_weights)
    synthetic_features=decoder.predict(noise)[2]

    classifier=VAE_attack_generator.build_classifier()
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
    (_,_),(_,_),names=format_data.get_rolled_data()
    noise=np.random.normal(size=(int(num_obs),latent_input_shape[0],latent_input_shape[1]))

    decoder=VAE_attack_generator.build_decoder()
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

    (X_clean,y_clean),(X,y),names=format_data.get_rolled_data()

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

    ind=np.shuffle(len(X))

    return (X[ind],y[ind])
