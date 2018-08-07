# Need to pass in: NUM_LAYS, NUM_RUNS, SIZE_BN, nn_lX, num_measurements

import numpy as np
import scipy
import os

path_in = '../data/raw_weights'
path_in = '/home/dave/cswdip/data_local/raw_w_for_musig_may4/'

num_measurements = [8000]#, 2000, 4000, 8000]
NUM_RUNS = 2

NUM_LAYS = 19
NUM_SAMPS = 64 # number of samples to take per file read
NUM_PULLS = 1024 # number of pulls comparing different runs

NUM_PULLS = 1
SIZE_BN = 64 # number of weights in batchnorm layers
RUN_INDICES = range(1,NUM_RUNS,1) # index for different runs of generated weights
conv_layers = [0,3,6,9,12,15,18]
nn_l1 = 32768 # 32x64x4x4, layer 1
nn_l2 = 147456 # 64x64x6x6, layers 2-6
nn_l7 = 1024 # 64x1x4x4, layer 7

def get_index_sets(NUM_SAMPS): 
	if NUM_SAMPS > 64:
		print('Error: Number of layer samples exceeds number of neurons in layer')
		return
	else:
		i_1  = np.random.choice(nn_l1, NUM_SAMPS)
		i_26 = np.random.choice(nn_l2, NUM_SAMPS)
		i_7  = np.random.choice(nn_l7, NUM_SAMPS)
		return i_1, i_26, i_7

def make_symmetric(matrix): # make matrix symmetric, i.e. sig_ij = sig_ji for all i not equal j
	for i in xrange(NUM_LAYS):
		for j in xrange(NUM_LAYS):
			if i != j:
				ij = matrix[i][j]
				ji = matrix[j][i]
				avg = .5*(ij+ji)
				matrix[i][j] = avg
				matrix[j][i] = avg
	return matrix

def add_lay_to_fn(fn, lay):
	return fn + str(lay) + '.npy'

def get_samps(fn, w_indices):
	arr = np.load(fn)
	arr_f = np.ndarray.flatten(arr) # flatten array so we can sample it
	  
	samps = np.random.choice(arr_f, NUM_SAMPS) # take samples from random weights in layer
	return samps

def get_one_mu_sig(nomenc): # choose at random two runs; get a mu, sigma corresponding to those two sets of weights
	run_ind = np.random.choice(RUN_INDICES,1)[0] # get one run index (i.e. from a run of DIP)
	fn = path_in + nomenc + str(run_ind) + '_lay' # add num_meas, run_ind to filename
	
	w_ind1, w_ind26, w_ind7 = get_index_sets(NUM_SAMPS) # get set of indices for higher consistency across layers

	M = np.zeros((NUM_LAYS, NUM_SAMPS)) # M: 7x1024 matrix containing all weights for this pull
	mu = np.zeros(NUM_LAYS)
	
	# construct M, mu
	for i in xrange(NUM_LAYS):
		lay = i+1
		fn_i = add_lay_to_fn(fn, lay) # add layer to filename string so we can load weight data
		w_ind = w_ind1 if lay == 1 else w_ind7 if lay == 7 else w_ind26 # get index set depending on layer i
		
		if i in conv_layers:
			w_i = get_samps(fn_i, w_ind)            
		else:
			w_i = np.load(fn_i)
			
		M[i] =  w_i       
		mu[i] = np.mean(w_i)
	sig = ((1/float(NUM_SAMPS))*(np.dot(M,M.T))) - np.outer(mu.T, mu) # calculate sig = (1/1024)*M*M^T - mu^T*mu
	return mu, sig  

def get_many_mu_sig_and_avg(nomenc): # average mu sig over NUM_PULLS
	sig = np.zeros((NUM_PULLS,NUM_LAYS,NUM_LAYS))
	mu = np.zeros((NUM_PULLS, NUM_LAYS))
	for k in xrange(NUM_PULLS):
		mu[k], sig[k] = get_one_mu_sig(nomenc) 
	#average over all different pulls
	mu = np.mean(mu,axis=0)
	sig = np.mean(sig,axis=0)
	
	sig_symm = make_symmetric(sig) # make matrix symmetric
	
	return mu, sig_symm
			
def get_mu_sig_over_num_meas(): # get mu, sig for each num_meas
	nm = len(num_measurements)
	sigs = np.zeros((nm, NUM_LAYS, NUM_LAYS))
	mus = np.zeros((nm, NUM_LAYS))
	
	for i in xrange(nm):
		nm_i = num_measurements[i]
		# nomenc is string, determines which num_meas file we are reading
		nomenc = 'meas' + str(nm_i) + '/' + 'meas' + str(nm_i) + '_run'
		mus[i], sigs[i] = get_many_mu_sig_and_avg(nomenc)
	return mus, sigs

def estimate_mu_sig_from_samples():
	NNM = len(num_measurements)
	mu_master = np.zeros((NUM_RUNS,NNM,NUM_LAYS))
	sig_master = np.zeros((NUM_RUNS,NNM,NUM_LAYS,NUM_LAYS))

	path_read = os.path.abspath('.') + '/data/disbn_samples/'

	for i in xrange(NUM_RUNS):
		mu_master[i] = np.load(path_read + 'mu_vecs' + str(i+1) + '.npy')
		sig_master[i] = np.load(path_read + 'sig_mats' + str(i+1) + '.npy')

	mu_avg = np.mean(mu_master,axis=0)
	sig_avg = np.mean(sig_master, axis=0)

	return mu_avg, sig_avg