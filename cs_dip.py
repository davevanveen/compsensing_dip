import numpy as np
import parser
import torch
from torch.autograd import Variable
import baselines

import utils
from utils import DCGAN_XRAY, DCGAN_MNIST


args = parser.parse_args('configs.json') # contains neural net hyperparameters

CUDA = torch.cuda.is_available()
dtype = utils.set_dtype(CUDA)

se = torch.nn.MSELoss(reduce=False).type(dtype)

BEGIN_CHECKPOINT = 50 # iteration at which to begin checking exit condition
EXIT_WINDOW = 25 # number of consecutive MSE values upon which we compare
NGF = 64
BATCH_SIZE = 1

meas_loss_ = np.zeros((args.NUM_RESTARTS, BATCH_SIZE))
reconstructions_ = np.zeros((args.NUM_RESTARTS, BATCH_SIZE, args.NUM_CHANNELS, \
                    args.IMG_SIZE, args.IMG_SIZE))

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
                
                if (i >= BEGIN_CHECKPOINT): # if optimzn has converged, exit descent
                    should_exit, loss_min_restart = utils.exit_check(loss_temp[-EXIT_WINDOW:],i)
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
