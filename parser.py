import argparse
import json
import utils


def parse_args(config_file='configs.json'):
    ALL_CONFIG = json.load(open(config_file))

    CONFIG = ALL_CONFIG["data_agnostic_configs"]

    # Set default values for data-agnostic configurations
    DEMO = CONFIG["demo"]
    DATASET = CONFIG["dataset"]
    ALG = CONFIG["alg"]
    NUM_ITER = CONFIG["num_iterations"]
    parser = argparse.ArgumentParser()

    parser.add_argument('--DATASET', type=str, default=DATASET,\
            help='dataset, DEFAULT=mnist. SUPPORTED=[mnist, xray, retino]')
    parser.add_argument('--ALG', nargs='+', type=str, default=ALG,\
            help='algorithm, DEFAULT=csdip. SUPPORTED=[csdip, dct, wavelet]. BM3D-AMP, TVAL3 must be run in Matlab.')
    parser.add_argument('--NUM_MEASUREMENTS', nargs='+', type=int, default = None, \
            help='number of measurements, DEFAULT dependent on dataset.')
    parser.add_argument('--DEMO', type=str, default=DEMO, \
            help='demo, boolean. Set True to run method over subset of 5 images \
             (default). Set False to run over entire dataset.')
    parser.add_argument('--NUM_ITER', type=int, default=NUM_ITER,\
    		help='number of iterative weight updates, DEFAULT=' + str(NUM_ITER))
    parser.add_argument('--NUM_RESTARTS', type=int, default= None,\
    		help='number of restarts, DEFAULT dependent on dataset.')
    
    args = parser.parse_args()

    # set values for data-specific configurations
    SPECIFIC_CONFIG = ALL_CONFIG[args.DATASET]
    args.IMG_SIZE = SPECIFIC_CONFIG["img_size"]
    args.NUM_CHANNELS = SPECIFIC_CONFIG["num_channels"]
    args.Z_DIM = SPECIFIC_CONFIG["z_dim"] #input seed
    args.LR_FOLDER = SPECIFIC_CONFIG["lr_folder"]

    # if data-specific arg not set by user
    if not args.NUM_MEASUREMENTS:
        args.NUM_MEASUREMENTS = SPECIFIC_CONFIG["num_measurements"]
    if not args.NUM_RESTARTS:
        args.NUM_RESTARTS = SPECIFIC_CONFIG["num_restarts"]

    utils.check_args(args) # check to make sure args are correct

    return args
