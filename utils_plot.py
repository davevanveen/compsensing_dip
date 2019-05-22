# package imports
import numpy as np
import argparse
import scipy.io as sio
import matplotlib.pyplot as plt
# user defined imports
import utils

class Args(object): # to create an empty argument class
    def __init__(self):
        pass

def construct_arg(args): # define dataset-specific arguments
	if args.DATASET == 'mnist':
	    args.IMG_SIZE = 28
	    args.NUM_CHANNELS = 1
	elif args.DATASET == 'xray':
	    args.IMG_SIZE = 256
	    args.NUM_CHANNELS = 1
	elif args.DATASET == 'retino':
	    args.IMG_SIZE = 128
	    args.NUM_CHANNELS = 3
	else:
	    raise NotImplementedErrorFalse
	    
	if args.DEMO == 'True':
	    args.IMG_PATH = 'data/{0}_demo/'.format(args.DATASET)
	    args.MSE_PLOT_PATH = 'plots/{0}_demo_mse.pdf'.format(args.DATASET)
	    args.REC_PLOT_PATH = 'plots/{0}_demo_reconstructions'.format(args.DATASET)
	else:
	    args.IMG_PATH = 'data/{0}/'.format(args.DATASET)
	    args.MSE_PLOT_PATH = 'plots/{0}_mse.pdf'.format(args.DATASET)
	    args.REC_PLOT_PATH = 'plots/{0}_reconstructions'.format(args.DATASET)

	return args

def renorm_bm3d(x): # maps [0,256] output from .mat file to [-1,1] for conistency 
    return .0078125*x - 1

def get_plot_data(dataloader, args): # load reconstructions and compute mse
	RECONSTRUCTIONS = dict()
	MSE = dict()
	for ALG in args.ALG_LIST:
	    args.ALG = ALG
	    RECONSTRUCTIONS[ALG] = dict()
	    MSE[ALG] = dict()

	    for NUM_MEASUREMENTS in args.NUM_MEASUREMENTS_LIST:
	        args.NUM_MEASUREMENTS = NUM_MEASUREMENTS
	        RECONSTRUCTIONS[ALG][NUM_MEASUREMENTS] = list()
	        MSE[ALG][NUM_MEASUREMENTS] = list()
	        
	        for _, (batch, _, im_path) in enumerate(dataloader):
	            if args.DATASET == 'retino':
	                batch_ = batch.numpy()[0]
	            else:
	                batch_ = batch.numpy()[0][0]
	            rec_path = utils.get_path_out(args,im_path)
	            if ALG == 'bm3d' or ALG == 'tval3':
	                rec = sio.loadmat(rec_path)['x_hat']
	                rec = renorm_bm3d(rec) # convert [0,255] --> [-1,1] 
	                if args.DATASET == 'retino':
	                    rec = np.transpose(rec, (2,0,1))                  
	            else:
	                rec = np.load(rec_path)
	            n = rec.ravel().shape[0]
	            mse = np.power(np.linalg.norm(batch_.ravel() - rec.ravel()),2)/(1.0*n)
	            RECONSTRUCTIONS[ALG][NUM_MEASUREMENTS].append(rec)
	            MSE[ALG][NUM_MEASUREMENTS].append(mse)
	return RECONSTRUCTIONS, MSE

def set_kwargs():
    KWARGS_DICT = {'csdip':{"fmt":'r-', "label":'Ours', "marker":"^", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1},
              'dct':{"fmt":'g-', "label":'Lasso-DCT', "marker":"s", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1, "markerfacecolor":'None'},
              'wavelet':{"fmt":'b-', "label":'Lasso-DB4', "marker":"D", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1, "markerfacecolor":'None'},
               'bm3d':{"fmt":'o-', "label":'BM3D-AMP', "marker":"D", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1, "markerfacecolor":'None'},
               'tval3':{"fmt":'v-', "label":'TVAL3', "marker":"D", "markersize":7,"capsize":4., "linewidth":1, "elinewidth":1, "markerfacecolor":'None'}
              }
    return KWARGS_DICT

def renorm(x):
    return 0.5*x + 0.5

def plot_format(y_lim, args):
    plt.ylim([0,y_lim])
    plt.ylabel('Reconstruction Error')
    plt.xlabel('Number of Measurements')
    plt.xticks(args.NUM_MEASUREMENTS_LIST,args.NUM_MEASUREMENTS_LIST, rotation=90)
    plt.legend(loc='upper right')

def plot_mse(mse_alg, args, kwargs):
    y_temp = []
    y_error = []
    x_temp = args.NUM_MEASUREMENTS_LIST
    for NUM_MEASUREMENTS in args.NUM_MEASUREMENTS_LIST:
        n = len(mse_alg[NUM_MEASUREMENTS])
        mse = np.mean(mse_alg[NUM_MEASUREMENTS])
        error = np.std(mse_alg[NUM_MEASUREMENTS]) / np.sqrt(1.0*n)
        y_temp.append(mse)
        y_error.append(error)
    # print(y_temp)
    plt.errorbar(x_temp,y_temp,y_error,**kwargs)

figure_height = 5
NUM_PLOT = 5

def set_axes(alg_name, ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.set_ylabel(alg_name, fontsize=14)

def frame_image(image, cmap = None):
    frame=plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frame = frame.imshow(image, cmap=cmap)

# plot images according to different data format (rgb/grayscale, original/reconstruction, from python/matlab)
def plot_image(image, args, flag):
	if flag == 'orig': # for plotting an original image
		if args.NUM_CHANNELS == 3: # for rgb images
			frame_image(renorm(image[0].cpu().numpy().transpose((1,2,0))))
		elif args.NUM_CHANNELS == 1: # for grayscale images
			frame_image(renorm(image[0].cpu().numpy().reshape((args.IMG_SIZE,args.IMG_SIZE))), cmap='gray')
		else:
			raise ValueError('NUM_CHANNELS must be 1 for grayscale or 3 for rgb images.')
	elif flag == 'recons': # for plotting reconstructions
		if args.NUM_CHANNELS == 3: # for rgb images
			if args.ALG == 'csdip':
				frame_image(renorm(image[0][0].transpose((1,2,0))))
			elif args.ALG == 'bm3d':
				frame_image(utils.renorm(np.asarray(image.transpose(1,2,0))))
			elif args.ALG == 'dct' or args.ALG == 'wavelet':
				frame_image(renorm(image.reshape(-1,128,3,order='F').swapaxes(0,1)))
			else:
				raise ValueError('Plotting rgb images is supported only by csdip, bm3d, dct, wavelet.')
		elif args.NUM_CHANNELS == 1: # for grayscale images
			frame_image(renorm(image.reshape(args.IMG_SIZE,args.IMG_SIZE)), cmap='gray')
		else:
			raise ValueError('NUM_CHANNELS must be 1 for grayscale or 3 for rgb images.')
	else:
		raise ValueError('flag input must be orig or recons for plotting original image or reconstruction, respectively.')


### UNUSED FUNCTIONS BELOW ###
# from PIL import Image
def classify(rgb_tuple):
    # will classify pixel into one of following categories based on manhattan distance
    colors = {"red": (255, 0, 0),
              "yellow": (255,255,0),
              "lyellow": (255,255,150),
              "orange": (255,165,0),
              "black": (0,0,0),
              "brown": (132, 85, 27),
              "obrown": (202, 134, 101),
              "bgreen": (12,136,115),
              "green" : (0,255,0),
              "purple": (128,0,128),
              "lpurple": (211,134,248)
              }
    manhattan = lambda x,y : abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2]) 
    distances = {k: manhattan(v, rgb_tuple) for k, v in colors.items()}
    color = min(distances, key=distances.get)
    return color

white = (255,255,255)
not_converged = ['green','purple','lpurple','bgreen']

def process_rgb(imarray): # update pixel values if not converged
    img = Image.fromarray((imarray*255).astype('uint8'), 'RGB')
    width,height = img.size
    for x in xrange(width):
        for y in xrange(height):
            r,g,b = img.getpixel((x,y))
            col = classify((r,g,b))
            if col in not_converged:
                img.putpixel((x,y), white)
                tt = np.array(img)
    return np.array(img)