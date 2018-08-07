import numpy as np
import parser
import torch
from torch.autograd import Variable

import utils as utils
from utils import DCGAN_XRAY
from utils import DCGAN_MNIST


args = parser.parse_args('configs.json')
# CS-DIP hyperparameters
LR = args.learning_rate
MOM = args.momentum
WD = args.weight_decay
NUM_ITER = args.number_iterations
NUM_RESTARTS = args.number_restarts
NGF = 64
# Data-specific parameters
Z_DIM = args.Z_DIM
DATASET = args.dataset
BATCH_SIZE = 1
NUM_CHANNELS = args.NUM_CHANNELS
IMG_SIZE = args.IMG_SIZE # pixel height/width, i.e. length of square image


CUDA = torch.cuda.is_available()

if CUDA : 
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    ltype = torch.LongTensor

se = torch.nn.MSELoss(reduce=False).type(dtype)

z = Variable(torch.zeros(BATCH_SIZE*Z_DIM).type(dtype).view(BATCH_SIZE,Z_DIM,1,1))
z.data.normal_().type(dtype)
z.requires_grad = False

mse_min = 0.
begin_checkpoint = 50 # iteration at which to begin checking exit condition
exit_window = 25 # number of consecutive MSE values upon which we compare

mse_ = np.zeros((NUM_RESTARTS,BATCH_SIZE))
meas_loss_ = np.zeros((NUM_RESTARTS,BATCH_SIZE))
reconstructions_ = np.zeros((NUM_RESTARTS, BATCH_SIZE, NUM_CHANNELS, IMG_SIZE, IMG_SIZE))

def csdip_optimize(num_measurements, batch, label):
	for j in range(NUM_RESTARTS):
	    # initialize network
	    if DATASET =='xray':
	        net = DCGAN_XRAY(Z_DIM,NGF,IMG_SIZE,NUM_CHANNELS,num_measurements) 
	    elif DATASET == 'mnist':
	        net = DCGAN_MNIST(Z_DIM,NGF,IMG_SIZE,NUM_CHANNELS,num_measurements)
	    net.fc.requires_grad = False
	    net.fc.weight.data = torch.Tensor((1/np.sqrt(1.0*num_measurements))*\
	    np.random.randn(num_measurements, IMG_SIZE*IMG_SIZE*NUM_CHANNELS)).type(dtype) # A[num_measurements]

	    reset_list = [net.conv1, net.bn1, net.conv2, net.bn2, net.conv3, net.bn3, \
	                  net.conv4, net.bn4, net.conv5]
	    for temp in reset_list:
	       temp.reset_parameters()

	    allparams = [temp for temp in net.parameters()]
	    allparams = allparams[:-1] # get rid of last item in list (fc layer)
	    

	    z = Variable(torch.zeros(BATCH_SIZE*Z_DIM).type(dtype).view(BATCH_SIZE,Z_DIM,1,1))
	    z.data.normal_().type(dtype)

	    if CUDA:
	        net.cuda()

	    batch, label = batch.type(dtype), label.type(ltype)
	    batch_measurements = Variable(torch.mm(batch.view(BATCH_SIZE,-1),net.fc.weight.data.permute(1,0)),\
	                             requires_grad=False) 

	    optim = torch.optim.RMSprop(allparams,lr=LR, momentum=MOM, weight_decay=WD)

	    mse_temp = []
	    for i in range(NUM_ITER):
	        # TODO: Move optimizer to utils file
	        optim.zero_grad()
	        loss = torch.mean(torch.sum(se(net.measurements(z,batch_size=BATCH_SIZE),batch_measurements),dim=1))

	        loss.backward()
	        mse_value = np.mean(se(net(z),Variable(batch,requires_grad=False)).data.cpu().numpy(), axis=(1,2,3))
	        mse_temp.append(mse_value[0])
	        
	        if (i >= begin_checkpoint): # if past a certain number of iterations
	            # check to see if we've obtained the minimum MSE
	            should_exit, mse_min_temp = utils.exit_check(mse_temp[-exit_window:],i)
	            if should_exit == True:
	                break
	        else:
	            should_exit = False

	        optim.step()

	    # if first 'restart', set mse_min OR if we've set a new low with future 'restart'
	    if (j == 0) or (j > 0 and mse_min_temp < mse_min): 
	        mse_min = mse_min_temp   

	    # get reconstructions
	    reconstructions_[j] = net(z).data.cpu().numpy()
	    # get mse of reconstructions
	    mse_[j] = mse_min_temp
	    # get measurement losses
	    meas_loss_[j] = np.sum(se(net.measurements(z,batch_size=BATCH_SIZE),batch_measurements).data.cpu().numpy(),axis=1)

	return reconstructions_, mse_, meas_loss_