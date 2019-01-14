import numpy as np
import parser
import torch
from torch.autograd import Variable
import baselines

import utils

args = parser.parse_args('configs.json') # contains neural net hyperparameters

CUDA = torch.cuda.is_available()
dtype = utils.set_dtype(CUDA)

se = torch.nn.MSELoss(reduction='none').type(dtype)

BATCH_SIZE = 1
EXIT_WINDOW = 21

meas_loss_ = np.zeros((args.NUM_RESTARTS, BATCH_SIZE))
reconstructions_ = np.zeros((args.NUM_RESTARTS, BATCH_SIZE, args.NUM_CHANNELS, \
                    args.IMG_SIZE, args.IMG_SIZE))


def dip_estimator(args):
    def estimator(A_val, y_batch_val, args):
        '''
        if args.LEARNED_REG:
            lr_mu_ = np.load(args.LEARNED_REG_MU_PATH)
            lr_sigma_ = np.load(args.LEARNED_REG_SIGMA_PATH)
            lr_mu = torch.FloatTensor(lr_mu_).type(dtype)
            lr_sigma_inv = torch.FloatTensor(np.linalg.inv(lr_sigma_)).type(dtype)
            def get_reg_loss(layers):
                layer_means = torch.cat([layer.mean().view(1) for layer in layers])
                #layers_means = torch.cat([layer.mean() for layer in layers])
                reg_loss = torch.matmul(layer_means-lr_mu,torch.matmul(lr_sigma_inv,layer_means-lr_mu))
                #reg = torch.matmul(layer_means.t - lr_mu)elayers_means - lr_mu)
                #reg_loss = torch.pow(reg.norm(),2)
                return reg_loss #se(reg_loss,Variable(torch.zeros(1).type(dtype),requires_grad=False))
        '''


        y = torch.FloatTensor(y_batch_val).type(dtype) # cast measurements to GPU if possible
        A = torch.FloatTensor(A_val).type(dtype)
               
        for j in range(args.NUM_RESTARTS):
            
            net = utils.init_dcgan(args) # initialize DCGAN based on dataset


            # set random seed value to get same z -- only use when tuning hparams
            #torch.manual_seed(1)
            #torch.cuda.manual_seed(1)

            z = torch.zeros(BATCH_SIZE*args.Z_DIM).type(dtype).view(BATCH_SIZE,args.Z_DIM,1,1)
            z.data.normal_().type(dtype)
            if CUDA:
                net.cuda()
            
#            optim = torch.optim.SGD(net.parameters(),lr=args.LR, momentum=args.MOM, weight_decay=args.WD)
            optim = torch.optim.RMSprop(net.parameters(),lr=args.LR, momentum=args.MOM, weight_decay=args.WD)
            #optim = torch.optim.Adam(net.parameters(),lr=args.LR)
            loss_temp = []

            # variables for saving recons of last 50 iterations
            recons_iter = [] 

            for i in range(args.NUM_ITER):

                optim.zero_grad()

                net_measurements = torch.matmul(net(z).view(BATCH_SIZE,-1),A)
                y_loss = torch.mean(torch.sum(se(net_measurements,y),dim=1))

                tv_loss = args.LMBD_TV* \
                        (torch.sum(torch.abs(net(z)[:, :, :, :-1] - net(z)[:, :, :, 1:]))\
                        +torch.sum(torch.abs(net(z)[:, :, :-1, :] - net(z)[:, :, 1:, :]))) 
                if args.LEARNED_REG:
                    lr_reg_loss = 10.*get_reg_loss(net.parameters()) 
                    total_loss = y_loss + tv_loss + lr_reg_loss 
                else:
                    total_loss = y_loss + tv_loss  
 
                '''
                if i >= args.NUM_ITER - EXIT_WINDOW:
                    recons_iter.append(net(z).data.cpu().numpy()) # save reconstr'n for that iter'n
                    loss_temp.append(total_loss.data.cpu().numpy()) # save loss value of each iteration to array
                    if i == args.NUM_ITER - 1: # if at last iteration in loop
                        iter_best = np.argmin(loss_temp)
                '''

                meas_loss = y_loss.data.cpu().numpy()
                if (i+1)%50==0:
                    print(meas_loss,(np.linalg.norm(net(z)[0][0].data.cpu().numpy().ravel()-args.x))**2/65536.) 
                total_loss.backward()
                optim.step()
        '''
            reconstructions_[j] = recons_iter[iter_best] # get best reconstruction over last 20 iterations       
            meas_loss_[j] = meas_loss
        
            
        idx_best = np.argmin(meas_loss_,axis=0) # index of restart with lowest loss
        x_hat = reconstructions_[idx_best] # choose best reconstruction from all restarts
        '''
        x_hat = net(z).data.cpu().numpy()
        return x_hat

    return estimator
