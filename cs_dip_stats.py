import numpy as np
import pickle as pkl
import os

import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch_dip_utils as utils
from dip.models.__init__ import get_net
from dip.utils.common_utils import get_noise
from dip.models.downsampler import Downsampler


BATCH_SIZE = 1
CUDA = torch.cuda.is_available()
GRAYSCALE = True
IMG_SIZE = 28 # pixel height. also pixel width
TRAIN = True # True if training, false otherwise

# CS-DIP hyperparameters
LR = 1e-3 # learning rate
MOM = 0.9 # momentum
NUM_ITER = 300 # number of iterations
WD = 1e-4 # weight decay
THRESH = 0.0
LOG_FREQ = 50 # log frequency
NUM_RESTARTS = 5 # number of restarts
Z_NUM = 128 # input seed
NGF = 64 # number of filters per layer


NUM_MEASUREMENTS_LIST = [15,35] # number of measurements to iterate over
NOISE_SDEV_LIST = [0] # standard deviations of noise to add
LAMBDA_LIST = [0] # weight to give the learned regularizer
BASIS = ['dip'] # method: 
DATASET = ['mnist'] # dataset: 'mnist' or 'xray'

if CUDA : 
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    ltype = torch.LongTensor

if GRAYSCALE:
    NUM_CHANNELS = 1
    compose = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5),(.5,.5,.5))
    ])
else:
    NUM_CHANNELS = 3 #rgb
    compose = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5),(.5,.5,.5))
    ])

dataloader = torch.utils.data.DataLoader(datasets.MNIST('../data/',download=True, \
				train=TRAIN,transform=compose), shuffle=False, batch_size=BATCH_SIZE)
# directory for saving generated weights
path_write = os.path.abspath('.') + '/data/raw_weights/' 

# initialize input seed z
z = Variable(torch.zeros(BATCH_SIZE*Z_NUM).type(dtype).view(BATCH_SIZE,Z_NUM,1,1))
z.data.normal_().type(dtype)
z.requires_grad = False

RECONSTRUCTIONS = dict()
MSE = dict()
MEASUREMENT_LOSS = dict()

mse_temp = dict()
for dset in DATASET:
    for basis in BASIS:
        for num_measurements in NUM_MEASUREMENTS_LIST:
            print('Number of measurements: ' + str(num_measurements))
            # check if the pickle files already exist. if they do, update the respective
            # empty dictionaries using contents of the files
            rec_f = '{0}_reconstructions_{1}_{2}.pkl'.format(dset,basis,num_measurements)
            mse_f = '{0}_mse_dip_{1}_{2}.pkl'.format(dset,basis,num_measurements)
            meas_f = '{0}_measurement_loss_{1}_{2}.pkl'.format(dset,basis,num_measurements)
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
                RECONSTRUCTIONS[num_measurements] = dict()
            if num_measurements in MSE.keys():
                pass
            else:
                MSE[num_measurements] = dict()
            if num_measurements in MEASUREMENT_LOSS.keys():
                pass
            else:
                MEASUREMENT_LOSS[num_measurements] = dict()
            for noise_sdev in NOISE_SDEV_LIST:
                if noise_sdev in RECONSTRUCTIONS[num_measurements].keys():
                    pass
                else:
                    RECONSTRUCTIONS[num_measurements][noise_sdev] = dict()
                if noise_sdev in MSE[num_measurements].keys():
                    pass
                else:
                    MSE[num_measurements][noise_sdev] = dict()
                if noise_sdev in MEASUREMENT_LOSS[num_measurements].keys():
                    pass
                else:
                    MEASUREMENT_LOSS[num_measurements][noise_sdev] = dict()
                
                # iterate over weighting for learned regularizer
                for lambda_ in LAMBDA_LIST :

                    # this is the final level, which holds a list of all the things we care about
                    if lambda_ in RECONSTRUCTIONS[num_measurements][noise_sdev].keys():
                        pass
                    else:
                        RECONSTRUCTIONS[num_measurements][noise_sdev][lambda_] = []
                    if lambda_ in MSE[num_measurements][noise_sdev].keys():
                        pass
                    else:
                        MSE[num_measurements][noise_sdev][lambda_] = []
                    if lambda_ in MEASUREMENT_LOSS[num_measurements][noise_sdev].keys():
                        pass
                    else:
                        MEASUREMENT_LOSS[num_measurements][noise_sdev][lambda_] = []
                    
                    mse_temp[lambda_] = []

                    for k, (batch, label) in enumerate(dataloader):
                        if k>=100:
                            break
                        
                        #direc = '/home/dave/cswdip/data_local/mnist_raw_w_for_musig_may8/meas' + str(num_measurements) + '/'
                        nomenc = 'meas' + str(num_measurements) + '_run' + str(k+1)      
                        
                        
                        mse_ = np.zeros((NUM_RESTARTS,BATCH_SIZE))
                        meas_loss_ = np.zeros((NUM_RESTARTS,BATCH_SIZE))
                        reconstructions_ = np.zeros((NUM_RESTARTS, BATCH_SIZE, NUM_CHANNELS, IMG_SIZE, IMG_SIZE))
                        mse_min = 0.
                        
                        new_min_counter = 0 # count new minimums
                        for j in range(NUM_RESTARTS):
                            # initialize network
                            if dset =='xray':
                                net = DCGAN_X(Z_NUM,NGF,IMG_SIZE,NUM_CHANNELS,num_measurements) 
                            elif dset == 'mnist':
                                net = DCGAN_MNIST(Z_NUM,64,IMG_SIZE,NUM_CHANNELS,num_measurements)
                            net.fc.requires_grad = False
                            net.fc.weight.data = torch.Tensor((1/np.sqrt(1.0*num_measurements))*\
                            np.random.randn(num_measurements, IMG_SIZE*IMG_SIZE*NUM_CHANNELS)).type(dtype) # A[num_measurements]

                            reset_list = [net.conv1, net.bn1, net.conv2, net.bn2, net.conv3, net.bn3, \
                                          net.conv4, net.bn4, net.conv5]
                            for temp in reset_list:
                               temp.reset_parameters()

                            allparams = [temp for temp in net.parameters()]
                            allparams = allparams[:-1] # get rid of last item in list (fc layer)
                            z = initialize_z(z,z_init)
      
                            # already initialized on cpu (above), now move to gpu 
                            if CUDA:
                                net.cuda()
      
                            batch, label = batch.type(dtype), label.type(ltype)
                            batch_measurements = Variable(torch.mm(batch.view(BATCH_SIZE,-1),net.fc.weight.data.permute(1,0))\
                                                    + noise_sdev * torch.randn(num_measurements).type(dtype),\
                                                     requires_grad=False) 

                            optim = torch.optim.RMSprop(allparams,lr=1e-3, momentum=0.9)#, weight_decay=WD)

                            mse_temp[lambda_] = []
                            for i in range(NUM_ITER):
                                optim.zero_grad()
                                meas_loss = torch.mean(torch.sum(se(net.measurements(z,batch_size=BATCH_SIZE),batch_measurements),dim=1))

                                if lambda_ > 0:
                                    ssq_iter = get_ssq_lay(allparams,mu_value)
                                    reg_loss = loss_learned_reg([1./c for c in np.diag(sigma_value)], ssq_iter)
                                else:
                                    reg_loss = 0.0

                                loss = meas_loss + lambda_*reg_loss
                                loss.backward()
                                mse_value = np.mean(se(net(z),Variable(batch,requires_grad=False)).data.cpu().numpy(), axis=(1,2,3))
                                mse_temp[lambda_].append(mse_value[0])

                                # want to save a numpy file of weights when we have lowest MSE across NUM_ITER + NUM_RESTARTS
                                if (j == 0 and mse_value[0] == min(mse_temp[lambda_])) \
                                    or (j > 0 and mse_value[0] < mse_min):
                                    new_min_counter += 1
                                    w = [x for x in net.parameters()]
                                    w = w[:-1] # get rid of last fc layer

                                    for m in xrange(len(w)):
                                        #ind = lay_oi[m] # index of layer within allparams (0, 3, 6, ..., 18)
                                        lay = m+1 # layer we're looking at (0, 1, ..., 7)
                                        lw = w[m].data.cpu().numpy() # layer weights in np array

                                        fn = path_write + nomenc + '_lay' + str(lay) + '.npy'
                                        np.save(fn, lw)   
                                
                                if (i >= begin_checkpoint): # if past a certain number of iterations
                                    # check to see if we've obtained the minimum MSE
                                    should_exit, mse_min_temp = exit_check(mse_temp[lambda_][-exit_window:],k,j)
                                    if should_exit == True:
                                        break
                                else:
                                    should_exit = False

                                optim.step()

			                # if first 'restart', set mse_min OR if we've set a new low with future 'restart'
                            if (j == 0) or (j > 0 and mse_min_temp < mse_min): 
                                mse_min = mse_min_temp #but what if we never get mse_min_temp?    

                            # get reconstructions
                            reconstructions_[j] = net(z).data.cpu().numpy()
                            # get mse of reconstructions
                            mse_[j] = mse_min_temp
#                             mse_[j] = np.mean(se(net(z),Variable(batch,requires_grad=False)).data.cpu().numpy(), axis=(1,2,3))
                            # get measurement losses
                            meas_loss_[j] = np.sum(se(net.measurements(z,batch_size=BATCH_SIZE),batch_measurements).data.cpu().numpy(),axis=1)

                        idx = np.argmin(mse_,axis=0)
                        # append the above values to the appropriate dictionaries
                        for j in range(BATCH_SIZE):
                            RECONSTRUCTIONS[num_measurements][noise_sdev][lambda_].append(reconstructions_[idx[j],j])
                            MSE[num_measurements][noise_sdev][lambda_].append(mse_[idx[j],j])
                            MEASUREMENT_LOSS[num_measurements][noise_sdev][lambda_].append(meas_loss_[idx[j],j])
                            print(k,mse_[idx[j],j])
                        
            # pickle the dictionaries
            with open(rec_f,'w') as f:
                pkl.dump(RECONSTRUCTIONS, f)
            with open(mse_f,'w') as f:
                pkl.dump(MSE, f)
            with open(meas_f,'w') as f:
                pkl.dump(MEASUREMENT_LOSS, f)