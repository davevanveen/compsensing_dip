import numpy as np
import parser
import torch
from torch.autograd import Variable
import baselines

import utils
from utils import DCGAN_XRAY, DCGAN_MNIST


args = parser.parse_args('configs.json') # contains neural net hyperparameters
NGF = 64
BATCH_SIZE = 1

CUDA = torch.cuda.is_available()

if CUDA : 
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    ltype = torch.LongTensor

se = torch.nn.MSELoss(reduce=False).type(dtype)

mse_min = 0.
begin_checkpoint = 50 # iteration at which to begin checking exit condition
exit_window = 25 # number of consecutive MSE values upon which we compare

meas_loss_ = np.zeros((args.NUM_RESTARTS, BATCH_SIZE))
reconstructions_ = np.zeros((args.NUM_RESTARTS, BATCH_SIZE, args.NUM_CHANNELS, \
                    args.IMG_SIZE, args.IMG_SIZE))

# Notes
    # Nested functions? No special purpose. Just so we can get an 'estimator' f'n and see x-hat created in main
    # Batch size / how to deal with batches - treat as 1 image; hard code BATCH_SIZE
    # MSE calculated in separate plotting file from reading reconstructions
    # Create file hierarchy with layers for (1) dataset (2) basis and (3) NUM_MEAS 
    # Anything in dip_estimator we can cut down?



# TODO: file hierarchy!!! (see above)
def dip_estimator(args):
    def estimator(A_val, y_batch_val, args):

        y = Variable(torch.Tensor(y_batch_val).type(dtype)) # cast measurements to GPU if possible

        for j in range(args.NUM_RESTARTS):
            if args.DATASET == 'xray':
                net = DCGAN_XRAY(args.Z_DIM, NGF, args.IMG_SIZE,\
                    args.NUM_CHANNELS, args.NUM_MEASUREMENTS)
            elif args.DATASET == 'mnist':
                net = DCGAN_MNIST(args.Z_DIM, NGF, args.IMG_SIZE,\
                    args.NUM_CHANNELS, args.NUM_MEASUREMENTS)  

            net.fc.requires_grad = False
            net.fc.weight.data = torch.Tensor(A_val.T) # set A to be fc layer


            # We don't need this, correct?
            # reset_list = [net.conv1, net.bn1, net.conv2, net.bn2, net.conv3, net.bn3, \
            #               net.conv4, net.bn4, net.conv5] # dataset-dependent
            # for temp in reset_list:
            #    temp.reset_parameters()

            allparams = [temp for temp in net.parameters()]
            allparams = allparams[:-1] # get rid of last item in list (fc layer)
            
            z = Variable(torch.zeros(BATCH_SIZE*args.Z_DIM).type(dtype).view(BATCH_SIZE,args.Z_DIM,1,1))
            z.data.normal_().type(dtype)
            z.requires_grad = False

            if CUDA:
                net.cuda()

            optim = torch.optim.RMSprop(allparams,lr=args.LR, momentum=args.MOM, weight_decay=args.WD)

            loss_temp = [] 
            for i in range(args.NUM_ITER):

                optim.zero_grad()
                loss = torch.mean(torch.sum(se(net.measurements(z,batch_size=BATCH_SIZE),y),dim=1))
                loss.backward()
                
                meas_loss = np.sum(se(net.measurements(z,batch_size=BATCH_SIZE),y).data.cpu().numpy(),axis=1)
                loss_temp.append(meas_loss) # save loss value of each iteration to array
                
                if (i >= begin_checkpoint): # if optimzn has converged, exit descent
                    should_exit, loss_min_restart = utils.exit_check(loss_temp[-exit_window:],i)
                    if should_exit == True:
                        meas_loss = loss_min_restart # get first loss value of exit window
                        break
                else:
                    should_exit = False

                optim.step()  

            reconstructions_[j] = net(z).data.cpu().numpy() # get reconstructions        
            meas_loss_[j] = meas_loss # get last measurement loss for a given restart

        idx_best = np.argmin(meas_loss_,axis=0) # index of restart with lowest loss
        x_hat = reconstructions_[idx_best] # choose best reconstruction from all restarts

        return x_hat

    return estimator
