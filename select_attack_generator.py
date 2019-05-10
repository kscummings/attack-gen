'''
select best loss weight
'''
from sklearn.model_selection import train_test_split

import os
import numpy as np
import pandas as pd

import format_data
import VAE_attack_generator
from VAE_attack_generator import train_vae

'''
CONSTANTS
'''

NUM_TIMES=2
EPOCHS=5
BATCH_SIZE=256
SEE_FROM=0
WANT_TO_SEE=False
TEST_SIZE=0.3
MOTHER_DIR="loss_weight_selection"
SEARCH_LIST=[2]#[0,0.1,0.2,0.25,0.3333,0.5,1,2]

'''
SEARCH
'''

def search(data,gen_weight,get_baseline=True):
    '''
    given list of generator loss weights, train VAE and record final losses
    inputs
    #   data=(X_tr,y_tr,X_val,y_val)
    #   gen_weight=list of gen weights to try
    #   get_baseline=do a trial to train VAE by itself w/o classification arm
    '''
    (X_tr,y_tr,X_val,y_val)=data
    os.makedirs(MOTHER_DIR,exist_ok=True)

    # save data
    data_dir=os.path.join(MOTHER_DIR, "data")
    os.makedirs(data_dir,exist_ok=True)
    np.save(os.path.join(data_dir,"train"),X_tr)
    np.save(os.path.join(data_dir,"test"),X_val)
    np.save(os.path.join(data_dir,"train_response"),y_tr)
    np.save(os.path.join(data_dir,"test_response"),y_val)

    # start results
    temp = (np.zeros((len(gen_weight)+1,8)) if get_baseline else np.zeros((len(gen_weight),6)))
    results=pd.DataFrame(temp,columns=['gen_weight','class_weight','loss','val_loss',
        'class_loss','val_class_loss','gen_loss','val_gen_loss'])

    # look at VAE trained on its own
    if get_baseline:
        print("getting baseline")
        output_dir=os.path.join(MOTHER_DIR,"vae_baseline")
        train_vae(data=(X_tr,y_tr,X_val,y_val),
                  output_dir=output_dir,
                  gen_weight=1,
                  classification_weight=0,
                  checkins=NUM_TIMES,
                  epochs=EPOCHS,
                  see_from=SEE_FROM,
                  want_to_see=WANT_TO_SEE,
                  batch_size=BATCH_SIZE)
        res=pd.read_csv(os.path.join(output_dir,"loss_results.csv"))
        results.iloc[0][:2]=np.array([1,0])
        results.iloc[0][2:]=np.array(res.iloc[len(res)-1])
        results.to_csv(os.path.join(MOTHER_DIR,"gen_weight_search.csv"), index=False) # write so far

    # search for best gen weight
    for g in np.arange(len(gen_weight)):
        print("On trial for gen weight {}".format(gen_weight[g]))
        output_dir=os.path.join(MOTHER_DIR,"gen_weight_{}".format(gen_weight[g]))
        train_vae(data=(X_tr,y_tr,X_val,y_val),
                  output_dir=output_dir,
                  gen_weight=gen_weight[g],
                  checkins=NUM_TIMES,
                  epochs=EPOCHS,
                  see_from=SEE_FROM,
                  want_to_see=WANT_TO_SEE,
                  batch_size=BATCH_SIZE)

        res=pd.read_csv(os.path.join(output_dir,"loss_results.csv"))
        results.iloc[g+1][:2]=np.array([gen_weight[g],1])
        results.iloc[g+1][2:]=np.array(res.iloc[len(res)-1])
        results.to_csv(os.path.join(MOTHER_DIR,"gen_weight_search.csv"), index=False) # write so far


def main():
    # control for dataset
    (_,_),(X,y),names=format_data.get_rolled_data()
    X_tr,X_val,y_tr,y_val=train_test_split(X,y,stratify=y,test_size=TEST_SIZE,shuffle=True)
    search((X_tr,y_tr,X_val,y_val),SEARCH_LIST)

if __name__ == '__main__':
    main()
