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
print(args)
DATASET = args.dataset
DATA_DIR = args.DATA_DIR
NET_PATH = args.NET_PATH
NUM_CHANNELS = args.NUM_CHANNELS
IMG_SIZE = args.IMG_SIZE # pixel height/width, i.e. length of square image


NUM_MEASUREMENTS = args.NUM_MEASUREMENTS # TODO: cycle this over a list, add user input
NUM_MEASUREMENTS_LIST = [NUM_MEASUREMENTS] # number of measurements to iterate over

BATCH_SIZE = 1

BASIS = ['dip'] # method:  put this in config so they can loop through list


compose = utils.define_compose(NUM_CHANNELS, IMG_SIZE)
# DOWNLOAD 100 MNIST IMAGES AND SAVE IN REPO, don't use Torch's MNIST loader
# dataloader = torch.utils.data.DataLoader(datasets.MNIST('../data/',download=True, \
# 				train=TRAIN,transform=compose), shuffle=False, batch_size=BATCH_SIZE)
dataset = datasets.ImageFolder('data/tomo/', transform = compose)
# TODO: shuffle data with more than one image
dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)


RECONSTRUCTIONS = dict()
MSE = dict() # CALUCATE LATER IN PLOT FILE

#A = baselines.get_A(args.IMG_SIZE*args.IMG_SIZE*args.NUM_CHANNELS, num_measurements)
#y = np.dot(x,A)

for basis in BASIS: # DO WE NEED THIS?
    '''
    if basis == 'csdip':
        estimator = cs_dip.dip_estimator(args)
    elif basis == 'dct':
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


        # rec_f = 'reconstructions/{0}/{1}'.format(basis,num_measurements) also dataset

        # save x_hat in numpy file? nested directories OR pickle files
            # save numpy files in nested directories since hierarchy is simple
            # ISSUE: need filenames of images; or just save with index created based upon batch
            # fix file

        if os.path.exists(rec_f):
            with open(rec_f,'rb') as f:
                temp = pkl.load(f)
                RECONSTRUCTIONS.update(temp)


        if num_measurements in RECONSTRUCTIONS.keys():
            pass
        else:
            RECONSTRUCTIONS[num_measurements] = []

        for k, (batch, label) in enumerate(dataloader):
            if k>=100:
                break            

            # DIP specific, e.g. if basis == 'dip'
            reconstructions_, mse_, meas_loss_ = csd.csdip_optimize(num_measurements, batch, label)
            idx = np.argmin(mse_,axis=0)



            for j in range(BATCH_SIZE):
                RECONSTRUCTIONS[num_measurements].append(reconstructions_[idx[j],j])

                print('Image ' + str(k+1))
                    
        # pickle the dictionaries
        with open(rec_f,'w') as f:
            pkl.dump(RECONSTRUCTIONS, f)

        print('Done!')
