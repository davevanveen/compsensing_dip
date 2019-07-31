import numpy as np
import pickle as pkl
import os
import parser
import numpy as np

import torch
from torchvision import datasets

import utils
import cs_dip
import baselines as baselines 
import time

NEW_RECONS = False

args = parser.parse_args('configs.json')
print(args)

NUM_MEASUREMENTS_LIST, ALG_LIST = utils.convert_to_list(args)
NOISE_LIST = [0, 0.1, 1, 5, 10, 20, 50, 100]

dataloader = utils.get_data(args) # get dataset of images

for num_meas in NUM_MEASUREMENTS_LIST:
    args.NUM_MEASUREMENTS = num_meas 
    
    # init measurement matrix
    A = baselines.get_A(args.IMG_SIZE*args.IMG_SIZE*args.NUM_CHANNELS, args.NUM_MEASUREMENTS)
    
    for _, (batch, _, im_path) in enumerate(dataloader):

        for noise in NOISE_LIST:
            args.NOISE = noise # set value to induce noise
            mmm = args.NUM_MEASUREMENTS 
            # second argument is for np.random.normal is \sigma, i.e. std devn
            # thus feed in \sigma / sqrt(m) to get variance of \sigma^2 / m
            eta = np.random.normal(0, noise /  np.sqrt(mmm) , mmm)
        

            x = batch.view(1,-1).cpu().numpy() # define image
            y = np.dot(x,A) + eta

            for alg in ALG_LIST:
                args.ALG = alg

                if utils.recons_exists(args, im_path): # to avoid redundant reconstructions
                    continue
                NEW_RECONS = True

                if alg == 'csdip':
                    estimator = cs_dip.dip_estimator(args)
                elif alg == 'dct':
                    estimator = baselines.lasso_dct_estimator(args)
                elif alg == 'wavelet':
                    estimator = baselines.lasso_wavelet_estimator(args)
                elif alg == 'bm3d' or alg == 'tval3':
                    raise NotImplementedError('BM3D-AMP and TVAL3 are implemented in Matlab. \
                                            Please see GitHub repository for details.')
                else:
                    raise NotImplementedError

                x_hat = estimator(A, y, args)

                utils.save_reconstruction(x_hat, args, im_path)

if NEW_RECONS == False:
    print('Duplicate experiment configurations. No new data generated.')
else:
    print('Reconstructions generated!')
