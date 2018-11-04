import numpy as np
import parser
import torch
from torch.autograd import Variable
import baselines

import utils
from utils import DCGAN_XRAY, DCGAN_MNIST

print(torch.__version__)
args = parser.parse_args('configs.json') # contains neural net hyperparameters

CUDA = torch.cuda.is_available()
dtype = utils.set_dtype(CUDA)

se = torch.nn.MSELoss(reduction='none').type(dtype)

BEGIN_CHECKPOINT = 50 # iteration at which to begin checking exit condition
EXIT_WINDOW = 25 # number of consecutive MSE values upon which we compare
NGF = 64
BATCH_SIZE = 1

meas_loss_ = np.zeros((args.NUM_RESTARTS, BATCH_SIZE))
reconstructions_ = np.zeros((args.NUM_RESTARTS, BATCH_SIZE, args.NUM_CHANNELS, \
                    args.IMG_SIZE, args.IMG_SIZE))

def dip_estimator(args):
    def estimator(A_val, y_batch_val, args):

        y = torch.FloatTensor(y_batch_val).type(dtype) # cast measurements to GPU if possible
        A = torch.FloatTensor(A_val).type(dtype)
        for j in range(args.NUM_RESTARTS):
            if args.DATASET == 'xray':
                net = DCGAN_XRAY(args.Z_DIM, NGF, args.IMG_SIZE,\
                    args.NUM_CHANNELS, args.NUM_MEASUREMENTS)
            elif args.DATASET == 'mnist':
                net = DCGAN_MNIST(args.Z_DIM, NGF, args.IMG_SIZE,\
                    args.NUM_CHANNELS, args.NUM_MEASUREMENTS)  

            z = torch.zeros(BATCH_SIZE*args.Z_DIM).type(dtype).view(BATCH_SIZE,args.Z_DIM,1,1)
            z.data.normal_().type(dtype)
            if CUDA:
                net.cuda()
            
            optim = torch.optim.RMSprop(net.parameters(),lr=args.LR, momentum=args.MOM, weight_decay=args.WD)
            loss_temp = [] 
            for i in range(args.NUM_ITER):

                optim.zero_grad()
                net_measurements = torch.matmul(net(z).view(BATCH_SIZE,-1),A)
                tv_loss = 1e-1 * \
                        (torch.sum(torch.abs(net(z)[:, :, :, :-1] - net(z)[:, :, :, 1:]))\
                        +torch.sum(torch.abs(net(z)[:, :, :-1, :] - net(z)[:, :, 1:, :])))
                y_loss = torch.mean(torch.sum(se(net_measurements,y),dim=1)) 
                total_loss = y_loss + tv_loss
                total_loss.backward()
                optim.step()  

                #pixel_loss = np.mean(np.power(net(z).view(1,-1).cpu().detach().numpy() - args.x,2))
                meas_loss = y_loss.data.cpu().numpy()
                loss_temp.append(meas_loss) # save loss value of each iteration to array
               
                '''
                if (i >= BEGIN_CHECKPOINT): # if optimzn has converged, exit descent
                    should_exit, loss_min_restart = utils.exit_check(loss_temp[-EXIT_WINDOW:],i)
                    if should_exit == True:
                        meas_loss = loss_min_restart # get first loss value of exit window
                        break
                else:
                    should_exit = False
                '''

            reconstructions_[j] = net(z).data.cpu().numpy() # get reconstructions        
            meas_loss_[j] = meas_loss #np.mean(loss_temp[-20:]) # get last measurement loss for a given restart

        idx_best = np.argmin(meas_loss_,axis=0) # index of restart with lowest loss
        x_hat = reconstructions_[idx_best] # choose best reconstruction from all restarts

        return x_hat

    return estimator
