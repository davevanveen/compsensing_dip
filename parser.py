import argparse
import json
import utils


def parse_args(config_file='configs.json'):
    ALL_CONFIG = json.load(open(config_file))

    CONFIG = ALL_CONFIG["data_agnostic_configs"]

    # Set default values for data-agnostic configurations
    DATASET = CONFIG["dataset"]
    BASIS = CONFIG["basis"]
    LR = CONFIG["learning_rate"]
    MOM = CONFIG["momentum"]
    WD = CONFIG["weight_decay"]
    NUM_ITER = CONFIG["number_iterations"]
    NUM_RESTARTS = CONFIG["number_restarts"]
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATASET', type=str, default=DATASET,\
            help='dataset, DEFAULT=mnist. SUPPORTED=[mnist, xray]')
    parser.add_argument('--BASIS', nargs='+', type=str, default=BASIS,\
        help='basis, DEFAULT=csdip. SUPPORTED=[csdip, dct, wavelet]')
    parser.add_argument('--LR', type=float, default=LR,\
    		help='learning rate, DEFAULT=' + str(LR))
    parser.add_argument('--MOM', type=float, default=MOM,\
    		help='RMSProp momentum hyperparameter, DEFAULT=' + str(MOM))
    parser.add_argument('--WD', type=float, default=WD,\
    		help='l2 weight decay hyperparameter, DEFAULT=' + str(WD))
    parser.add_argument('--NUM_ITER', type=int, default=NUM_ITER,\
    		help='number of iterative weight updates, DEFAULT=' + str(NUM_ITER))
    parser.add_argument('--NUM_RESTARTS', type=int, default=NUM_RESTARTS,\
    		help='number of restarts, DEFAULT=' + str(NUM_RESTARTS))
    parser.add_argument('--NUM_MEASUREMENTS', nargs='+', type=int, default = None, \
            help='number of measurements, DEFAULT dependent on dataset')
	
    # add parser argument so that they can input num_measurements in list format
    # space separated after --
    # look at Bora's code

    args = parser.parse_args()
    
    print(args)

    # set values for data-specific configurations
    SPECIFIC_CONFIG = ALL_CONFIG[args.DATASET]
    args.NET_PATH = SPECIFIC_CONFIG["network_path"]
    args.DATA_DIR = SPECIFIC_CONFIG["data_dir"]
    args.IMG_SIZE = SPECIFIC_CONFIG["img_size"]
    args.NUM_CHANNELS = SPECIFIC_CONFIG["num_channels"]
    args.Z_DIM = SPECIFIC_CONFIG["z_dim"] #input seed

    NUM_MEAS_DEFAULT = SPECIFIC_CONFIG["num_measurements"]
    if not args.NUM_MEASUREMENTS:
        args.NUM_MEASUREMENTS = NUM_MEAS_DEFAULT

    utils.check_args(args) # check to make sure args are correct

    return args