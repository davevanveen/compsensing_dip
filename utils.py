import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

CUDA = torch.cuda.is_available()

if CUDA : 
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    ltype = torch.LongTensor

se = torch.nn.MSELoss(reduce=False).type(dtype)
l2 = nn.MSELoss().type(dtype)

class DCGAN_XRAY(nn.Module):
    def __init__(self, nz, ngf=64, output_size=256, nc=3, num_measurements=1000):
        super(DCGAN_X, self).__init__()
        self.nc = nc
        self.output_size = output_size

        self.conv1 = nn.ConvTranspose2d(nz, ngf, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf)
        self.conv2 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf)
        self.conv3 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf)
        self.conv4 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.conv5 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn5 = nn.BatchNorm2d(ngf)
        self.conv6 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn6 = nn.BatchNorm2d(ngf)
        self.conv7 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False) #output is image
    
        self.fc = nn.Linear(output_size*output_size*nc,num_measurements, bias=False) #output is A
        # each entry should be drawn from a Gaussian
        # don't compute gradient of self.fc
    
    def forward(self, x):
        input_size = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.tanh(self.conv7(x,output_size=(-1,self.nc,self.output_size,self.output_size)))
       
        return x

    def measurements(self, x, batch_size=1):
        # this gives the image
        # make it a single row vector of appropriate length
        y = self.forward(x).view(batch_size,-1)

        #passing it through the fully connected layer
        # returns A*image
        return self.fc(y)
    
    def measurements_(self, x):
        # measure an image x
        y = x.view(1,-1)
        return self.fc(y)

class DCGAN_MNIST(nn.Module):
    def __init__(self, nz, ngf=128, output_size=28, nc=1, num_measurements=10):
        super(DCGAN_MNIST, self).__init__()
        self.nc = nc
        self.output_size = output_size

        self.conv1 = nn.ConvTranspose2d(nz, ngf*8, 2, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf*8)
        self.conv2 = nn.ConvTranspose2d(ngf*8, ngf*4, 4, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*4)
        self.conv3 = nn.ConvTranspose2d(ngf*4, ngf*2, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*2)
        self.conv4 = nn.ConvTranspose2d(ngf*2, ngf, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.conv5 = nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False) 
        
        self.fc = nn.Linear(output_size*output_size*nc,num_measurements, bias=False) #output is A
        # each entry should be drawn from a Gaussian
        # don't compute gradient of self.fc

    def forward(self, x):
        input_size = x.size()
        x = F.upsample(F.relu(self.bn1(self.conv1(x))),scale_factor=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.upsample(F.relu(self.bn3(self.conv3(x))),scale_factor=2)
        x = F.upsample(F.relu(self.bn4(self.conv4(x))),scale_factor=2)
        x = F.tanh(self.conv5(x,output_size=(-1,self.nc,self.output_size,self.output_size)))
       
        return x
   


    def measurements(self, x, batch_size=1):
        # this gives the image
        # make it a single row vector of appropriate length
        y = self.forward(x).view(batch_size,-1)

        #passing it through the fully connected layer
        # returns A*image
        return self.fc(y)
    
    def measurements_(self, x):
        # measure an image x
        y = x.view(1,-1)
        return self.fc(y)

def norm(x):
    return x*2.0 - 1.0

def renorm(x):
    return 0.5*x + 0.5

def plot(x,renormalize=True):
    if renormalize:
        plt.imshow(renorm(x).data[0].cpu().numpy(), cmap='gray')
    else:
        plt.imshow(x.data[0].cpu().numpy(), cmap='gray')

# def initialize_z(z,value):
#     z.data = value.data.type(dtype)
#     return z

exit_window = 25 # number of consecutive MSE values upon which we compare
thresh_ratio = 20 # number of MSE values that must be larger for us to exit

def exit_check(window, i): # if converged, then exit current experiment
    mse_base = window[0] # get first mse value in window
    
    if len(np.where(window > mse_base)[0]) >= thresh_ratio: # if 20/25 values in window are higher than mse_base
        return True, mse_base
    else:
        mse_last = window[exit_window-1] #get the last value of MSE in window
        return False, mse_last


def define_compose(NC, IMG_SIZE): # define compose based on NUM_CHANNELS, IMG_SIZE
    if NC == 1: #grayscale
        compose = transforms.Compose([
            transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((.5,.5,.5),(.5,.5,.5))
        ])
    elif NC == 3: #rgb
        compose = transforms.Compose([
            transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((.5,.5,.5),(.5,.5,.5))
        ])
    return compose