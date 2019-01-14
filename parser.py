import argparse
import json
import utils


def parse_args(config_file='configs.json'):
    ALL_CONFIG = json.load(open(config_file))

    CONFIG = ALL_CONFIG["data_agnostic_configs"]

    # Set default values for data-agnostic configurations
    DEMO = CONFIG["demo"]
    DATASET = CONFIG["dataset"]
    BASIS = CONFIG["basis"]
    MEASUREMENT = CONFIG["measurement"]
    LR = CONFIG["learning_rate"]
    MOM = CONFIG["momentum"]
    WD = CONFIG["weight_decay"]
    NUM_ITER = CONFIG["number_iterations"]
    NUM_RESTARTS = CONFIG["number_restarts"]
    LMBD = CONFIG["lambda"] # lasso hparam
    LMBD_TV = CONFIG["lambda_tv"]
    OPTIMIZER = CONFIG["optimizer"]
    MODEL = CONFIG["model"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--DEMO', type=str, default=DEMO, \
            help='demo, boolean. Set True to run method over subset of 5 images \
             (default). Set False to run over entire dataset.')
    parser.add_argument('--DATASET', type=str, default=DATASET,\
            help='dataset, DEFAULT=mnist. SUPPORTED=[mnist, xray, retino]')
    parser.add_argument('--BASIS', nargs='+', type=str, default=BASIS,\
        help='basis, DEFAULT=csdip. SUPPORTED=[csdip, dct, wavelet, lasso-fourier]')
    parser.add_argument('--MEASUREMENT', type=str, default=MEASUREMENT,\
        help='measurement matrix, DEFAULT=gaussian. SUPPORTED=[gaussian, fourier]')
    parser.add_argument('--LR', type=float, default=LR,\
    		help='learning rate, DEFAULT=' + str(LR))
    parser.add_argument('--LMBD_TV', type=float, default=LMBD_TV,\
    		help='hyperparam for TV regularization, DEFAULT='+str(LMBD_TV))
    parser.add_argument('--MOM', type=float, default=MOM,\
    		help='RMSProp momentum hyperparameter, DEFAULT=' + str(MOM))
    parser.add_argument('--WD', type=float, default=WD,\
    		help='l2 weight decay hyperparameter, DEFAULT=' + str(WD))
    parser.add_argument('--LMBD', type=float, default=LMBD,\
    		help='lambda for L1, DEFAULT=' + str(LMBD))
    parser.add_argument('--NUM_ITER', type=int, default=NUM_ITER,\
    		help='number of iterative weight updates, DEFAULT=' + str(NUM_ITER))
    parser.add_argument('--OPTIMIZER', type=str, default=OPTIMIZER,\
            help='optimizer, DEFAULT=rms. SUPPORTED=[rms, adam]')
    parser.add_argument('--MODEL', type=str, default=MODEL,\
            help='generative model, DEFAULT=dcgan. SUPPORTED=[dcgan, unet]')
    parser.add_argument('--NUM_RESTARTS', type=int, default=NUM_RESTARTS,\
    		help='number of restarts, DEFAULT=' + str(NUM_RESTARTS))
    parser.add_argument('--NUM_MEASUREMENTS', nargs='+', type=int, default = None, \
            help='number of measurements, DEFAULT dependent on dataset')
    boolean_parser = parser.add_mutually_exclusive_group(required=False)
    boolean_parser.add_argument('--LEARNED_REG', dest='LEARNED_REG', action='store_true')
    boolean_parser.add_argument('--NO_LEARNED_REG', dest='LEARNED_REG', action='store_false')
    parser.set_defaults(LEARNED_REG=False)
    
    args = parser.parse_args()

    # set values for data-specific configurations
    SPECIFIC_CONFIG = ALL_CONFIG[args.DATASET]
    args.IMG_SIZE = SPECIFIC_CONFIG["img_size"]
    args.NUM_CHANNELS = SPECIFIC_CONFIG["num_channels"]
    args.Z_DIM = SPECIFIC_CONFIG["z_dim"] #input seed
    args.LR_FOLDER = SPECIFIC_CONFIG["lr_folder"]
    NUM_MEAS_DEFAULT = SPECIFIC_CONFIG["num_measurements"]
    if not args.NUM_MEASUREMENTS:
        args.NUM_MEASUREMENTS = NUM_MEAS_DEFAULT

    #print(args)

    utils.check_args(args) # check to make sure args are correct

    return args
