from sklearn.linear_model import Lasso
import scipy.fftpack as fftpack
import pywt
import copy
import numpy as np
import utils
import os
from scipy import io

def solve_lasso(A_val, y_val, lmbd=1e-1):
    num_measurements = y_val.shape[0]
    lasso_est = Lasso(alpha=lmbd)#,tol=1e-4,selection='random')
    lasso_est.fit(A_val.T, y_val.reshape(num_measurements))
    x_hat = lasso_est.coef_
    x_hat = np.reshape(x_hat, [-1])
    return x_hat

def dct2(image_channel):
    return fftpack.dct(fftpack.dct(image_channel.T, norm='ortho').T, norm='ortho')

def idct2(image_channel):
    return fftpack.idct(fftpack.idct(image_channel.T, norm='ortho').T, norm='ortho')

def db4(image_channel):
    coeffs = pywt.wavedec2(image_channel,'db4')
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    return arr, coeff_slices

def idb4(image_channel, coeff_slices):
    coeffs_from_arr = pywt.array_to_coeffs(image_channel, coeff_slices, output_format='wavedec2')
    return pywt.waverec2(coeffs_from_arr,'db4')

def vec(channels,num_channels):
    shape = channels[0].shape
    image = np.zeros((num_channels, shape[0], shape[1]))
    for i, channel in enumerate(channels):
        image[i, :, :] = channel
    return image.reshape([-1])

def devec(vector,num_channels):
    size = int(np.sqrt(vector.shape[0]/num_channels))
    image = np.reshape(vector, [num_channels, size, size])
    channels = [image[i, :, :] for i in range(num_channels)]
    return channels

def lasso_fourier_estimator(args):  #pylint: disable = W0613
    """ LASSO with fourier coefficients"""
    def estimator(A_val, y_batch_val, args):
        A_new = copy.deepcopy(A_val)

        y_val = y_batch_val[0]
        x_hat = solve_lasso(A_new, y_val, args.LMBD)
        x_hat = np.maximum(np.minimum(x_hat, 1), -1)
        return x_hat
    return estimator

def lasso_dct_estimator(args):  #pylint: disable = W0613
    """LASSO with DCT"""
    def estimator(A_val, y_batch_val, args):
        # One can prove that taking 2D DCT of each row of A,
        # then solving usual LASSO, and finally taking 2D ICT gives the correct answer.
        A_new = copy.deepcopy(A_val)
	for i in range(A_val.shape[1]):
            A_new[:, i] = vec([dct2(channel) for channel in devec(A_new[:, i],args.NUM_CHANNELS)],args.NUM_CHANNELS)

        y_val = y_batch_val[0]
        z_hat = solve_lasso(A_new, y_val, args.LMBD)
        x_hat = vec([idct2(channel) for channel in devec(z_hat,args.NUM_CHANNELS)],args.NUM_CHANNELS).T
        x_hat = np.maximum(np.minimum(x_hat, 1), -1)
        return x_hat
    return estimator

def lasso_wavelet_estimator(args):  #pylint: disable = W0613
    """LASSO with DWT"""
    def estimator(A_val, y_batch_val, args):
        # One can prove that taking 2D DWT of each row of A,
        # then solving usual LASSO, and finally taking 2D IWT gives the correct answer.
        A_new = copy.deepcopy(A_val)
        arr, coeff_slices = db4(devec(A_new[:,0],args.NUM_CHANNELS)[0])
        A_wav = np.zeros((args.NUM_CHANNELS*arr.shape[0]*arr.shape[1],A_val.shape[1]))
        for i in range(A_val.shape[1]):
            A_wav[:, i] = vec([db4(channel)[0] for channel in devec(A_new[:, i],args.NUM_CHANNELS)],args.NUM_CHANNELS)
        y_val = y_batch_val[0]
        z_hat = solve_lasso(A_wav, y_val, args.LMBD)
        x_hat = vec([idb4(channel,coeff_slices) for channel in devec(z_hat,args.NUM_CHANNELS)],args.NUM_CHANNELS).T
        x_hat = np.maximum(np.minimum(x_hat, 1), -1)
        return x_hat
    return estimator



def get_A(dimension,num_measurements,args):
    #np.random.seed(1) # set random seed value to get same A -- only use when tuning hparams
    if args.MEASUREMENT == 'gaussian':
        return np.sqrt(1.0/num_measurements)*np.random.randn(dimension,num_measurements)
    
    elif args.MEASUREMENT == 'fourier':
        filename = './measurement_matrices/fourier_{0}.npy'.format(num_measurements)
        filename_mat = './measurement_matrices/fourier_{0}.mat'.format(num_measurements)
        if os.path.exists(filename):
            A = np.load(filename)
        else:
            A = utils.fourier_measurements(num_measurements, args.IMG_SIZE)
            if not os.path.exists(os.path.dirname(filename)):
                os.mkdir(os.path.dirname(filename))
            np.save(filename,A)
	    A_dict = {'M': A.T}
	    io.savemat(filename_mat, A_dict)
        return A


