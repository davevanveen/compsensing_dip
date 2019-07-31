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

# above, NOISE_LIST were all \sigma's. 
# Now, VAR_LIST is \sigma^2; manually enter NOISE_LIST as sqrt of that to be int or 2 decimals
#\sigma^2 = [0, 1, 10, 50, 100] #update as of Wed 12p 
#NOISE_LIST = [0,1,3.16,7.07,10] # manually entered as sqrt of VAR_LIST
NOISE_LIST = [0,0.707,1,2.24,3.16,7.07,10,22.36,31.62]

dataloader = utils.get_data(args) # get dataset of images

for num_meas in NUM_MEASUREMENTS_LIST:
    args.NUM_MEASUREMENTS = num_meas 
    
    # init measurement matrix
    A = baselines.get_A(args.IMG_SIZE*args.IMG_SIZE*args.NUM_CHANNELS, args.NUM_MEASUREMENTS)
    for _, (batch, _, im_path) in enumerate(dataloader):    
        
        for noise in NOISE_LIST:
            args.NOISE = noise # set value to induce noise
            # mmm = args.NUM_MEASUREMENTS 
            # second argument is for np.random.normal is \sigma, i.e. std devn
            # thus feed in \sigma / sqrt(m) to get variance of \sigma^2 / m
            
            #eta = np.random.normal(0, noise /  np.sqrt(mmm) , mmm)
        

            x = batch.view(1,-1).cpu().numpy() # define image
            #y = np.dot(x,A) + eta

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

                # x_hat = estimator(A, y, args)
                x_hat = estimator(x, A, args)

                utils.save_reconstruction(x_hat, args, im_path)

if NEW_RECONS == False:
    print('Duplicate experiment configurations. No new data generated.')
else:
    print('Reconstructions generated!')
