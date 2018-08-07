import argparse
import json


def parse_args(config_file='config_json'):
    ALL_CONFIG = json.load(open(config_file))

    

    CONFIG = ALL_CONFIG["data_agnostic_configs"]
    

    # Set default values for data-agnostic configurations
    DATASET = CONFIG["dataset"]
    LR = CONFIG["learning_rate"]
    MOM = CONFIG["momentum"]
    WD = CONFIG["weight_decay"]
    NUM_ITER = CONFIG["number_iterations"]
    NUM_RESTARTS = CONFIG["number_restarts"]
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=DATASET,\
            help='dataset, DEFAULT=mnist. SUPPORTED=[mnist, xray]')
    parser.add_argument('--learning-rate', type=float, default=LR,\
    		help='learning rate, DEFAULT=' + str(LR))
    parser.add_argument('--momentum', type=float, default=MOM,\
    		help='RMSProp momentum hyperparameter, DEFAULT=' + str(MOM))
    parser.add_argument('--weight-decay', type=float, default=WD,\
    		help='l2 weight decay hyperparameter, DEFAULT=' + str(WD))
    parser.add_argument('--number-iterations', type=int, default=NUM_ITER,\
    		help='number of iterative weight updates, DEFAULT=' + str(NUM_ITER))
    parser.add_argument('--number-restarts', type=int, default=NUM_RESTARTS,\
    		help='number of restarts, DEFAULT=' + str(NUM_RESTARTS))

    # add parser argument so that they can input num_measurements in list format
    # space separated after --
    # look at Bora's code

    args = parser.parse_args()

    # set values for data-specific configurations
    SPECIFIC_CONFIG = ALL_CONFIG[args.dataset]
    args.NET_PATH = SPECIFIC_CONFIG["network_path"]
    args.DATA_DIR = SPECIFIC_CONFIG["data_dir"]
    args.IMG_SIZE = SPECIFIC_CONFIG["img_size"]
    args.NUM_CHANNELS = SPECIFIC_CONFIG["num_channels"]
    args.NUM_MEASUREMENTS = SPECIFIC_CONFIG["num_measurements"]
    args.Z_DIM = SPECIFIC_CONFIG["z_dim"] #input seed

    parser.add_argument('--number-measurements', type=int, \
    		default=args.NUM_MEASUREMENTS, \
			help='number of restarts, DEFAULT='+str(args.NUM_MEASUREMENTS))

    return args

    # GOAL: Give user option of defining num_measurements as input (int or list format)
    # ISSUE: Need dataset declared before we know the default value for NUM_MEAS
