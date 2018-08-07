import numpy as np
import os


from stats_utils import get_mu_sig_over_num_meas as get
from stats_utils import estimate_mu_sig_from_samples as est

# number of samples to generate statistics. Estimate accuracy increases with NUM_RUNS
NUM_RUNS = 2500
NUM_RUNS = 2
path = os.path.abspath('.') # current absolute path

path_samps = path + '/data/disbn_samples/' # directory to save samples
# generate samples of distribution statistics over the network weights
for i in xrange(NUM_RUNS):
	mus, sigs = get()
	np.save(path_samps + 'mu_vecs' + str(i+1) + '.npy', mus)
	np.save(path_samps + 'sig_mats' + str(i+1) + '.npy', sigs)

# average the sample statistics to get a final estimate for mu, sigma
mu, sig = est()
np.save(path + '/data/mu.npy', mu) # save to file
np.save(path + '/data/sig.npy', sig)

print('Statistics for network weights estimated.')




