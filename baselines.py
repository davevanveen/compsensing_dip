from sklearn.linear_model import Lasso
import scipy.fftpack as fftpack
import pywt
import copy
import numpy as np

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

def vec(channels):
    shape = channels[0].shape
    image = np.zeros((shape[0], shape[1], args.NUM_CHANNELS))
    for i, channel in enumerate(channels):
        image[:, :, i] = channel
    return image.reshape([-1])

def devec(vector):
    size = int(np.sqrt(vector.shape[0]))/args.NUM_CHANNELS
    image = np.reshape(vector, [size, size, args.NUM_CHANNELS])
    channels = [image[:, :, i] for i in range(args.NUM_CHANNELS)]
    return channels

def lasso_dct_estimator(args):  #pylint: disable = W0613
    """LASSO with DCT"""
    def estimator(A_val, y_batch_val, args):
        # One can prove that taking 2D DCT of each row of A,
        # then solving usual LASSO, and finally taking 2D ICT gives the correct answer.
        A_new = copy.deepcopy(A_val)
        for i in range(A_val.shape[1]):
            A_new[:, i] = vec([dct2(channel) for channel in devec(A_new[:, i])])

        x_hat_batch = []
        for j in range(args.BATCH_SIZE):
            y_val = y_batch_val[j]
            z_hat = solve_lasso(A_new, y_val, args.LMBD)
            x_hat = vec([idct2(channel) for channel in devec(z_hat)]).T
            x_hat = np.maximum(np.minimum(x_hat, 1), -1)
            x_hat_batch.append(x_hat)
        return x_hat_batch
    return estimator

def lasso_wavelet_estimator(args):  #pylint: disable = W0613
    """LASSO with DWT"""
    def estimator(A_val, y_batch_val, args):
        # One can prove that taking 2D DWT of each row of A,
        # then solving usual LASSO, and finally taking 2D IWT gives the correct answer.
        A_new = copy.deepcopy(A_val)
        arr, coeff_slices = db4(devec(A_new[:,0])[0])
        A_wav = np.zeros((arr.shape[0]*arr.shape[1],A_val.shape[1]))
        for i in range(A_val.shape[1]):
            A_wav[:, i] = vec([db4(channel)[0] for channel in devec(A_new[:, i])])
        x_hat_batch = []
        for j in range(batch_size):
            y_val = y_batch_val[j]
            z_hat = solve_lasso(A_wav, y_val, args.LMBD)
            x_hat = vec([idb4(channel,coeff_slices) for channel in devec(z_hat)]).T
            x_hat = np.maximum(np.minimum(x_hat, 1), -1)
            x_hat_batch.append(x_hat)
        return x_hat_batch
    return estimator

def get_A(dimension,num_measurements):
    return np.sqrt(1.0/num_measurements)*np.random.randn(dimension,num_measurements)
