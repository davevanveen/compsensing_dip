import numpy as np
import pickle as pkl
import os
import parser

import torch
from torchvision import datasets

import utils
import cs_dip
import baselines_python as baselines 
import time

NEW_RECONS = False

args = parser.parse_args('configs.json')
print(args)

NUM_MEASUREMENTS_LIST, BASIS_LIST = utils.convert_to_list(args)

dataloader = utils.get_data(args) # get dataset of images over which to iterate

for num_measurements in NUM_MEASUREMENTS_LIST:

    args.NUM_MEASUREMENTS = num_measurements
    A = baselines.get_A(args.IMG_SIZE*args.IMG_SIZE*args.NUM_CHANNELS, args.NUM_MEASUREMENTS)

    for _, (batch, _, im_path) in enumerate(dataloader):

        x = batch.view(1,-1).cpu().numpy() #for larger batch, change first arg of .view()
        y = np.dot(x,A)

        for basis in BASIS_LIST:

            args.BASIS = basis

            if utils.recons_exists(args, im_path): # if reconstruction exists for a given config
                continue
            NEW_RECONS = True

            if basis == 'csdip':
                estimator = cs_dip.dip_estimator(args)
            elif basis == 'dct':
                estimator = baselines.lasso_dct_estimator(args)
            elif basis == 'wavelet':
                estimator = baselines.lasso_wavelet_estimator(args)
            else:
                raise NotImplementedError

            utils.save_reconstruction(x_hat, args, im_path)

if NEW_RECONS == False:
    print('Duplicate reconstruction configurations. No new data generated.')
else:
    print('Reconstructions generated!')