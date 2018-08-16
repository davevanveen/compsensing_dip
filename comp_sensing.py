import numpy as np
import pickle as pkl
import os
import parser

import torch
from torchvision import datasets

import cs_dip
import utils
import baselines

args = parser.parse_args('configs.json')


NUM_MEASUREMENTS_LIST = args.NUM_MEASUREMENTS # TODO: cycle this over a list, add user input
if isinstance(args.NUM_MEASUREMENTS, int): # if looking at single value for NUM_MEAS, cast to list
    NUM_MEASUREMENTS_LIST = [NUM_MEASUREMENTS_LIST]

BATCH_SIZE = 1
#BASIS = ['csdip'] # method:  put this in config so they can loop through list


compose = utils.define_compose(args.NUM_CHANNELS, args.IMG_SIZE)
# DOWNLOAD 100 MNIST IMAGES AND SAVE IN REPO, don't use Torch's MNIST loader
# dataloader = torch.utils.data.DataLoader(datasets.MNIST('../data/',download=True, \
# 				train=TRAIN,transform=compose), shuffle=False, batch_size=BATCH_SIZE)
dataset = datasets.ImageFolder('data/tomo/', transform = compose)
# TODO: shuffle data with more than one image
dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)




for num_measurements in NUM_MEASUREMENTS_LIST:
    
    args.NUM_MEASUREMENTS = num_measurements
    A = baselines.get_A(args.IMG_SIZE*args.IMG_SIZE*args.NUM_CHANNELS, args.NUM_MEASUREMENTS)

    for _, (batch, _) in enumerate(dataloader):

        x = batch.view(BATCH_SIZE, -1).cpu().numpy()
        y = np.dot(x,A)

        for basis in args.BASIS: # check functionality for different bases
            # print(args.BASIS)
            # print(basis)
            
            if basis == 'csdip':
                estimator = cs_dip.dip_estimator(args)
            elif basis == 'dct':
                estimator = baselines.lasso_dct_estimator(args)
            elif basis == 'wavelet':
                estimator = baselines.lasso_wavelet_estimator(args)
            else:
                raise NotImplementedError

            x_hat = estimator(A,y,args)

            utils.save_reconstruction(x_hat, args)

print('Done!')
