import numpy as np
import pickle as pkl
import os
import parser

import torch
from torchvision import datasets

import cs_dip as csd
import utils
import baselines

args = parser.parse_args('configs.json')
DATASET = args.dataset
DATA_DIR = args.DATA_DIR
NET_PATH = args.NET_PATH
NUM_CHANNELS = args.NUM_CHANNELS
IMG_SIZE = args.IMG_SIZE # pixel height/width, i.e. length of square image


NUM_MEASUREMENTS = args.NUM_MEASUREMENTS # TODO: cycle this over a list, add user input
NUM_MEASUREMENTS_LIST = [NUM_MEASUREMENTS] # number of measurements to iterate over

BATCH_SIZE = 1

BASIS = ['dip'] # method:  put this in config so they can loop through list
# line 21 pass in args. estimator is a dictionary of functions by model type
# estimator is entire optimizn process for an algorithm. Pass in A, y, hparams

# ISSUE: Not sure how Ajil did baselines. Check with him so it's consistent across methods


compose = utils.define_compose(NUM_CHANNELS, IMG_SIZE)
# SHOULD WE HAVE THEM DOWNLOAD MNIST FROM OUR REPO, OR USE THE TORCH COMMAND?
# dataloader = torch.utils.data.DataLoader(datasets.MNIST('../data/',download=True, \
# 				train=TRAIN,transform=compose), shuffle=False, batch_size=BATCH_SIZE)
dataset = datasets.ImageFolder('data/tomo/', transform = compose)
# TODO: shuffle data with more than one image
dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)


RECONSTRUCTIONS = dict()
MSE = dict()
MEASUREMENT_LOSS = dict()

for basis in BASIS: # DO WE NEED THIS?
    '''
    A = baselines.get_A(args.IMG_SIZE*args.IMG_SIZE*args.NUM_CHANNELS, num_measurements)
    y = np.dot(x,A)
    if basis == 'dct':
        estimator = baselines.lasso_dct_estimator(args)
    elif basis == 'wavelet':
        estimator = baselines.lasso_wavelet_estimator(args)
    else:
        raise NotImplementedError
    x_hat = estimator(A,y,args)
    '''
    for num_measurements in NUM_MEASUREMENTS_LIST: #NEED TO ADD LIST FUNCTIONALITY LATER
        print('Number of measurements: ' + str(num_measurements))
        # check if the pickle files already exist. if they do, update the respective
        # empty dictionaries using contents of the files
        rec_f = '{0}_reconstructions_{1}.pkl'.format(basis,num_measurements)
        mse_f = '{0}_mse_dip_{1}.pkl'.format(basis,num_measurements)
        meas_f = '{0}_measurement_loss_{1}.pkl'.format(basis,num_measurements)
        if os.path.exists(rec_f):
            with open(rec_f,'rb') as f:
                temp = pkl.load(f)
                RECONSTRUCTIONS.update(temp)
        if os.path.exists(mse_f):
            with open(mse_f,'rb') as f:
                temp = pkl.load(f)
                MSE.update(temp)
        if os.path.exists(meas_f):
            with open(meas_f,'rb') as f:
                temp = pkl.load(f)
                MEASUREMENT_LOSS.update(temp)
        
        # check if the dictionary for num_measurements exists. if it does not,
        # create an empty dictionary in its place
        if num_measurements in RECONSTRUCTIONS.keys():
            pass
        else:
            RECONSTRUCTIONS[num_measurements] = []
        if num_measurements in MSE.keys():
            pass
        else:
            MSE[num_measurements] = []
        if num_measurements in MEASUREMENT_LOSS.keys():
            pass
        else:
            MEASUREMENT_LOSS[num_measurements] = []


        for k, (batch, label) in enumerate(dataloader):
            if k>=100:
                break            






            # DIP specific, e.g. if basis == 'dip'
            reconstructions_, mse_, meas_loss_ = csd.csdip_optimize(num_measurements, batch, label)
            idx = np.argmin(mse_,axis=0)






            # append the above values to the appropriate dictionaries
            for j in range(BATCH_SIZE):
                RECONSTRUCTIONS[num_measurements].append(reconstructions_[idx[j],j])
                MSE[num_measurements].append(mse_[idx[j],j])
                MEASUREMENT_LOSS[num_measurements].append(meas_loss_[idx[j],j])
                print('Image ' + str(k+1) + ', MSE: ' + str(mse_[idx[j],j]))
                    
        # pickle the dictionaries
        with open(rec_f,'w') as f:
            pkl.dump(RECONSTRUCTIONS, f)
        with open(mse_f,'w') as f:
            pkl.dump(MSE, f)
        with open(meas_f,'w') as f:
            pkl.dump(MEASUREMENT_LOSS, f)

        print('Done!')
