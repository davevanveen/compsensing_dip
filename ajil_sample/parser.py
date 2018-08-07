import argparse
import json


def parse_args(config_file='config_json'):
    ALL_CONFIG = json.load(open(config_file))
    
    DATASET = ALL_CONFIG["attacks"]["dataset"]
    
    CONFIG = ALL_CONFIG["attacks"]
    ADV_ITERS = CONFIG["adv_iters"]
    EARLY_STOP = CONFIG["early_stop"]
    ADV_THRESH = CONFIG["adv_thresh"]
    BATCH_SIZE = CONFIG["batch_size"]
    ATTACK_TYPE = CONFIG["attack_name"]
    EPS = CONFIG["epsilon"]
    L2_EPS = CONFIG["max_admissible_l2"]
    NUM_PGD_STEPS = CONFIG["pgd_steps"]
    PGD_LR = CONFIG["pgd_lr"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=DATASET,\
            help='dataset, DEFAULT=mnist. supported=[mnist, celebA]')
    parser.add_argument('--adv-iters', type=int, default=ADV_ITERS,\
            help='number of iterations for adversarial training. DEFAULT= 100000')
    esp = parser.add_mutually_exclusive_group(required=False)
    esp.add_argument('--early-stop', dest='early_stop', action='store_true')
    esp.add_argument('--no-early-stop', dest='early_stop', action='store_false')
    parser.set_defaults(early_stop=EARLY_STOP)
    parser.add_argument('--adv_thresh', type=float, default=ADV_THRESH,\
            help='')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,\
            help='batch size, DEFAULT=100')
    parser.add_argument('--attack-name', type=str, default=ATTACK_TYPE,\
            help='type of attack to perform, DEFAULT=latent_op. \
            supported=[latent_op, latent_normal, fgsm, ifgsm, pgd, cw]')
    parser.add_argument('--eps', type=float, default=EPS,\
            help='maximum adversarial perturbation allowed. DEFAULT=0.001')
    parser.add_argument('--l2-eps', type=float, default=L2_EPS,\
            help='maximum adversarial perturbation allowed in L2 norm. DEFAULT=0.005')
    parser.add_argument('--pgd-steps', type=int, default=NUM_PGD_STEPS,\
            help='number of steps for PGD/IFGSM. DEFAULT=500')
    parser.add_argument('--pgd-lr', type=float, default=PGD_LR,\
            help='learning rate for PGD/IFGSM. DEFAULT=0.1')

    rp = parser.add_mutually_exclusive_group(required=False)
    rp.add_argument('--attack-robust', dest='attack_robust', action='store_true')
    rp.add_argument('--attack-nonrobust', dest='attack_robust', action='store_false')
    parser.set_defaults(attack_robust=False)

    ip = parser.add_mutually_exclusive_group(required=False)
    ip.add_argument('--attack-inc', dest='attack_inc', action='store_true')
    ip.add_argument('--attack-vanilla', dest='attack_inc', action='store_false')
    parser.set_defaults(attack_inc=True)
    parser.add_argument('--save-dir', type=str, default="", \
            help='Setting this will save the classified images.')
     
    args = parser.parse_args()

    GLOBAL_CONFIG = ALL_CONFIG[args.dataset]
    args.Z_DIM = GLOBAL_CONFIG["z_dim"]
    args.GEN_PATH = GLOBAL_CONFIG["generator_path"]
    args.DATA_DIR = GLOBAL_CONFIG["data_dir"]
    args.INV_ITERS = GLOBAL_CONFIG["inversion_iters"]
    args.GAN_VARIANCE = GLOBAL_CONFIG["gan_variance"]
    args.GAN_MU = GLOBAL_CONFIG["gan_mu"]
    args.IMG_SHAPE = tuple(GLOBAL_CONFIG["img_shape"])
    args.TOTAL_SHAPE = GLOBAL_CONFIG["total_shape"]
    args.NUM_CLASSES = GLOBAL_CONFIG["num_classes"]
    
    args.CLA_PATH = GLOBAL_CONFIG["robust_classifier_path"] if \
        args.attack_robust else GLOBAL_CONFIG["classifier_path"]

    return args
