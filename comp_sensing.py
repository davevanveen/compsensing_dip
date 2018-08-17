import numpy as np
import pickle as pkl
import os
import parser

import torch
from torchvision import datasets

import utils
import cs_dip
import baselines

args = parser.parse_args('configs.json')

# if a single value entered for NUM_MEAS or BASIS, convert to list
NUM_MEASUREMENTS_LIST, BASIS_LIST = utils.convert_to_list(args)

BATCH_SIZE = 1

# class ImageFolderWithPaths(datasets.ImageFolder):
#     """Custom dataset that includes image file paths. Extends
#     torchvision.datasets.ImageFolder
#     """

#     # override the __getitem__ method. this is the method dataloader calls
#     def __getitem__(self, index):
#         # this is what ImageFolder normally returns 
#         original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
#         # the image file path
#         path = self.imgs[index][0]
#         # make a new tuple that includes original and the path
#         tuple_with_path = (original_tuple + (path,))
#         return tuple_with_path      


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

        for basis in BASIS_LIST: # check functionality for different bases
            
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
