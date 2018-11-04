import numpy as np
import os
import errno
import parser

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets,transforms

BATCH_SIZE = 1

class DCGAN_XRAY(nn.Module):
    def __init__(self, nz, ngf=64, output_size=256, nc=3, num_measurements=1000):
        super(DCGAN_XRAY, self).__init__()
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
    
    
    def forward(self, z):
        input_size = z.size()
        x = F.relu(self.bn1(self.conv1(z)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = torch.tanh(self.conv7(x,output_size=(-1,self.nc,self.output_size,self.output_size)))
       
        return x

class DCGAN_MNIST(nn.Module):
    def __init__(self, nz, ngf=64, output_size=28, nc=1, num_measurements=10):
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
        
    
    def forward(self, x):
        input_size = x.size()
        x = F.upsample(F.relu(self.bn1(self.conv1(x))),scale_factor=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.upsample(F.relu(self.bn3(self.conv3(x))),scale_factor=2)
        x = F.upsample(F.relu(self.bn4(self.conv4(x))),scale_factor=2)
        x = torch.tanh(self.conv5(x,output_size=(-1,self.nc,self.output_size,self.output_size)))
       
        return x
   
def norm(x):
    return x*2.0 - 1.0

def renorm(x):
    return 0.5*x + 0.5

def plot(x,renormalize=True):
    if renormalize:
        plt.imshow(renorm(x).data[0].cpu().numpy(), cmap='gray')
    else:
        plt.imshow(x.data[0].cpu().numpy(), cmap='gray')

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

def set_dtype(CUDA):
    if CUDA: # if cuda is available
        return torch.cuda.FloatTensor
    else:
        return torch.FloatTensor

def get_path_out(args, path_in):
    fn = path_leaf(path_in[0]) # format filename from path
    path_out = 'reconstructions/{0}/{1}/meas{2}/im{3}.npy'.format( \
            args.DATASET, args.BASIS, args.NUM_MEASUREMENTS, fn)
    full_path = os.getcwd()  + '/' + path_out
    return full_path


def recons_exists(args, path_in):
    path_out = get_path_out(args, path_in)
    print(path_out)
    if os.path.isfile(path_out):
        return True
    else:
        return False

# TODO: add filename of image to args so that multiple images can be written to same direc
def save_reconstruction(x_hat, args, path_in):

    path_out = get_path_out(args, path_in)


    if not os.path.exists(os.path.dirname(path_out)):
        try:
            os.makedirs(os.path.dirname(path_out))
        except OSError as exc: # guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    np.save(path_out, x_hat)

def check_args(args): # check args for correctness
    IM_DIMN = args.IMG_SIZE * args.IMG_SIZE * args.NUM_CHANNELS

    if isinstance(args.NUM_MEASUREMENTS, int):
        if args.NUM_MEASUREMENTS > IM_DIMN:
            raise ValueError('NUM_MEASUREMENTS must be less than image dimension ' \
                + str(IM_DIMN))
    else:
        for num_measurements in args.NUM_MEASUREMENTS:
            if num_measurements > IM_DIMN:
                raise ValueError('NUM_MEASUREMENTS must be less than image dimension ' \
                    + str(IM_DIMN))
    if not args.DEMO == 'False':
        if not args.DEMO == 'True':
            raise ValueError('DEMO must be either True or False.')

def convert_to_list(args): # returns list for NUM_MEAS, BATCH
    if not isinstance(args.NUM_MEASUREMENTS, list):
        NUM_MEASUREMENTS_LIST = [args.NUM_MEASUREMENTS]
    else:
        NUM_MEASUREMENTS_LIST = args.NUM_MEASUREMENTS
    if not isinstance(args.BASIS, list):
        BASIS_LIST = [args.BASIS]
    else:
        BASIS_LIST = args.BASIS
    return NUM_MEASUREMENTS_LIST, BASIS_LIST

def path_leaf(path):
    # if '/' in path and if '\\' in path:
    #     raise ValueError('Path to image cannot contain both forward and backward slashes')

    if '.' in path: # remove file extension
        path_no_extn = os.path.splitext(path)[0]
    else:
        raise ValueError('Filename does not contain extension')
    
    head, tail = os.path.split(path_no_extn)
    return tail or os.path.basename(head)

def get_data(args):

    compose = define_compose(args.NUM_CHANNELS, args.IMG_SIZE)

    if args.DEMO == 'True':
        image_direc = 'data/{0}_demo/'.format(args.DATASET)
    else:
        image_direc = 'data/{0}/'.format(args.DATASET)

    dataset = ImageFolderWithPaths(image_direc, transform = compose)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)

    return dataloader

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path      
