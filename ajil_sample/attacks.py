import os
import json
import numpy as np
import tensorflow as tf
from invert_and_classify import invert
from lib_external.cw_l2 import CarliniL2

from set_dataset import *

import scipy
from scipy.misc import imsave
from skimage import io
import parser
import numpy.linalg as la

if __name__ == "__main__":
    args = parser.parse_args('config.json')
    DATASET = args.dataset
    Z_DIM = args.Z_DIM
    GEN_PATH = args.GEN_PATH
    DATA_DIR = args.DATA_DIR
    INV_ITERS = args.INV_ITERS
    GAN_VARIANCE = args.GAN_VARIANCE
    GAN_MU = args.GAN_MU
    IMG_SHAPE = tuple(args.IMG_SHAPE)
    TOTAL_SHAPE = args.TOTAL_SHAPE
    NUM_CLASSES = args.NUM_CLASSES

    ADV_ITERS = args.adv_iters
    EARLY_STOP = args.early_stop
    ADV_THRESH = args.adv_thresh
    BATCH_SIZE = args.batch_size
    ATTACK_TYPE = args.attack_name
    EPS = args.eps
    L2_EPS = args.l2_eps
    NUM_PGD_STEPS = args.pgd_steps
    PGD_LR = args.pgd_lr
    CLA_PATH = args.CLA_PATH
    ATTACK_INC = args.attack_inc

    sess = tf.Session()
    x = tf.random_normal([BATCH_SIZE, Z_DIM])

    gen_func = get_generator(DATASET)
    cla_func = get_classifier(DATASET)
    inp_func = get_input(DATASET)

    out, gen_vars = gen_func(x, reuse=False)
    _, _, cla_vars = cla_func(tf.ones((BATCH_SIZE,) + IMG_SHAPE), reuse=False)

    # Restore Generator
    g_saver = tf.train.Saver(var_list=gen_vars)
    g_saver.restore(sess, GEN_PATH)

    # Restore Classifier
    c_saver = tf.train.Saver(var_list=cla_vars)
    if CLA_PATH[-1] == "/":
        c_saver.restore(sess, tf.train.latest_checkpoint(CLA_PATH))
    else:
        c_saver.restore(sess, CLA_PATH)


    ARGS = {"batch_size": BATCH_SIZE, "z_dim": Z_DIM, "inc_iters": INV_ITERS, "gan_mu":GAN_MU}

def remaining_initializer():
    all_vars = []
    if type(gen_vars) == dict:
        all_vars.extend(gen_vars.values())
    else:
        all_vars.extend(gen_vars)
    if type(cla_vars) == dict:
        all_vars.extend(cla_vars.values())
    else:
        all_vars.extend(cla_vars)
    return tf.variables_initializer(filter(lambda x:x not in all_vars, \
            tf.global_variables()))

def fgsm(eps, initial_im, labels):
    '''
    Function to perform FGSM attack
    
    Input:
        eps: inf norm of adversarial perturbation
        initial_im: input image
        labels: labels of input image
    
    Returns:
    initial_im, adversarial_im
    '''
    im_ = initial_im.copy()
    
    x_hat = tf.Variable(im_)

    logits, preds = cla_func(x_hat, reuse=True)
    labels_one_hot = tf.one_hot(labels,NUM_CLASSES)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)
    
    grad, = tf.gradients(loss, x_hat)
    adv_noise = eps*tf.sign(grad)
    adv = x_hat + adv_noise
    
    sess.run(tf.variables_initializer([x_hat]))
    adversarial_im, grad_ = sess.run([adv,grad]) 
    return initial_im, adversarial_im


def iterated_attack(eps, initial_im, labels, attack):
    '''
    Function to perform iterated attack (FGSM or PGD)
    
    Input:
        eps: inf norm of adversarial perturbation
        initial_im: input image
        labels: labels of input image
        
    
    Returns:
    initial_im, adversarial_im
    '''
    assert attack in ["pgd", "ifgsm"]
    im_ = initial_im.copy()
    lower = np.clip(initial_im - eps, 0, 1)
    upper = np.clip(initial_im + eps, 0, 1)
    SHAPE = (BATCH_SIZE, ) + IMG_SHAPE
     
    x = tf.placeholder(tf.float32, SHAPE)
    logits, preds = cla_func(x, reuse=True)

    labels_one_hot = tf.one_hot(labels,NUM_CLASSES)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, \
            labels=labels_one_hot)

    g, = tf.gradients(loss, x)
    
    orig_preds = sess.run(preds, {x: initial_im})
    for i in range(NUM_PGD_STEPS):
        g_ = sess.run(g, {x: im_})
        if attack == "ifgsm":
            g_ = np.sign(g_)
        im_ = im_ + PGD_LR*g_
        im_ = np.clip(im_, lower, upper)
        new_preds = sess.run(preds, {x: im_})
        acc = (orig_preds == new_preds).astype(np.float32).mean()
        print("Classifier Accuracy: %f" % (acc,))
    return initial_im, im_

def l2_pgd(eps, initial_im, labels):
    '''
    Function to perform L2 PGD
    
    Input:
        eps: l2 norm of adversarial perturbation
        initial_im: input image
        labels: labels of input image
        
    
    Returns:
    initial_im, adversarial_im
    '''
    im_ = initial_im.copy()
    SHAPE = (BATCH_SIZE, ) + IMG_SHAPE
     
    x = tf.placeholder(tf.float32, SHAPE)
    logits, preds = cla_func(x, reuse=True)

    labels_one_hot = tf.one_hot(labels,NUM_CLASSES)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, \
            labels=labels_one_hot)

    g, = tf.gradients(loss, x)
    
    orig_preds = sess.run(preds, {x: initial_im})
    for i in range(NUM_PGD_STEPS):
        g_ = sess.run(g, {x: im_})
        g_norm = la.norm(g_.reshape((BATCH_SIZE, -1)), axis=1)
        g_ = g_/g_norm[:, np.newaxis, np.newaxis, np.newaxis]
            
        im_ = im_ + PGD_LR*g_
        
        diff_ = im_ - initial_im
        diff_norm = np.maximum(eps, la.norm(diff_.reshape((BATCH_SIZE, -1)), axis=1))
        diff_ = diff_ * eps / diff_norm[:, np.newaxis, np.newaxis, np.newaxis]
        
        im_ = initial_im + diff_
        
        new_preds = sess.run(preds, {x: im_})
        acc = (orig_preds == new_preds).astype(np.float32).mean()
        print("Classifier Accuracy: %f" % (acc,))
    return initial_im, im_

def cw(initial_im, labels):
    class Model(object):
        def __init__(self):
            self.image_size = IMG_SHAPE[0]
            self.num_channels = IMG_SHAPE[2]
            self.num_labels = NUM_CLASSES

        def predict(self,img):
            logits, _ = cla_func(img, reuse=True)
            return logits

    TARGETED = False
    
    model = Model()
    
    labels_one_hot = tf.one_hot(labels,NUM_CLASSES)
    max_value = np.max(initial_im)
    min_value = np.min(initial_im)
    
    cwl2 = CarliniL2(sess, model, targeted=TARGETED, boxmin=min_value, boxmax=max_value)
    
    labels_ = sess.run(labels_one_hot)
    adversarial_im = np.float32(cwl2.attack(initial_im,labels_))
    
    return initial_im, adversarial_im

def bpda(eps, initial_im, initial_label):
    """
    BPDA attack from Athalye et al, used in attack on DefenseGAN from that
    paper. On the forward pass, we do the projection step, and on the backward
    pass we use the identity function, as specified in the paper.
    """
    lower = initial_im - eps
    upper = initial_im + eps
    adv = initial_im.copy()

    x = tf.placeholder(tf.float32, initial_im.shape)
    labels_one_hot = tf.one_hot(labels,NUM_CLASSES)

    z = tf.Variable(GAN_MU + GAN_VARIANCE*np.random.randn(BATCH_SIZE, Z_DIM).astype(np.float32))
    with tf.control_dependencies([tf.assign(z, np.zeros((BATCH_SIZE, Z_DIM)))]):
        loop, proj_loss = invert(z, gen_func, \
                                    initial_im, \
                                    ARGS, \
                                    reuse=True, \
                                    std=GAN_VARIANCE)
    with tf.control_dependencies([loop, proj_loss]):
        gen_img, _ = gen_func(z, reuse=True)
        logits, preds = cla_func(gen_img, reuse=True)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, \
                labels=labels_one_hot)
        g, = tf.gradients(loss, [gen_img])

    sess.run(tf.variables_initializer([z]))
    for i in range(NUM_PGD_STEPS):
        gi, new_preds = sess.run([gen_img, preds], {x: adv})
        acc = (initial_label == new_preds).astype(np.float32).mean()
        g_, l_= sess.run([g, loss], {x: adv})
        print("Acc: %f Loss: %f" % (acc,l_.mean()))
        g_ = np.sign(g_)
        adv = adv + PGD_LR*g_
        adv = np.clip(adv, lower, upper)
    return initial_im, im_

def latent_space_graph(z1, z2, lam, eps, z_vars, lam_lr, \
        gen_func, cla_func, total_shape, l2_eps):
    """
    Helper function for latent space attack, only creates graph operations. Only
    called once from latent_space_attack, but also called in a tf.while_loop in
    the training sequence, which is why it's in its own function.
    - z1: The first latent variable (could be fixed or a tf.Variable)
    - z2: The second latent variable (can only be a tf.Variable)
    - lam: Regularization penalty (to enforce hard L2 bound)
    - eps: Epsilon (mean squared error bound [translates indirectly to L2])
    - z_vars: An array of the z's we're allowed to modify (i.e. overpowered?)
    - lam_lr: Learning rate for lambda (set in config file)
    - gen_func, cla_func: generator and classifier  
    - total_shape: see TOTAL_SHAPE variable
    - l2_eps: see L2_EPS variable
    """
    # classification loss
    gen_img1, _ = gen_func(tf.identity(z1), reuse=True)
    gen_img2, _ = gen_func(tf.identity(z2), reuse=True)
    logits1, preds1 = cla_func(gen_img1, reuse=True)
    logits2, preds2 = cla_func(gen_img2, reuse=True)
    # want to maximize softmax cross-entropy distance between logits
    sm_logits1 = tf.nn.softmax(logits1)
    sm_logits2 = tf.nn.softmax(logits2)
    adv_loss = tf.reduce_mean(tf.reduce_sum( \
            sm_logits1*tf.log(sm_logits2) + sm_logits2*tf.log(sm_logits1), axis=1))
    # Current distance in generated images (in image space)
    image_diffs = tf.square(gen_img1 - gen_img2)
    image_diffs = tf.sqrt(tf.reduce_sum(image_diffs, axis=[1,2,3]))/total_shape
    # adversarial pct
    adversarial_ims = tf.where(tf.equal(preds1, preds2), \
                tf.zeros_like(preds1, tf.float32), \
                tf.ones_like(preds1, tf.float32))
    allowable_ims = tf.where(tf.less(image_diffs, l2_eps), \
                tf.ones_like(preds1, tf.float32), \
                tf.zeros_like(preds1, tf.float32))
    # tf.multiply acts as a logical AND:
    truly_adversarial = tf.multiply(adversarial_ims, allowable_ims)
    pct_adv = tf.reduce_mean(truly_adversarial)
    # lagrangian loss
    image_diff = tf.reduce_mean(tf.square(gen_img1 - gen_img2))
    lagrangian_loss = lam*(image_diff - eps)
    # z loss
    mean_loss = tf.reduce_mean(0.01*(tf.square(tf.reduce_mean(z1, axis=1)) +  \
                    tf.square(tf.reduce_mean(z2, axis=1))))
    # total_loss
    total_loss = adv_loss + lagrangian_loss #+ mean_loss
    # Average l2 distance of adversarial images
    diff_where_adv = tf.multiply(adversarial_ims, image_diffs)
    avg_l2 = tf.reduce_sum(diff_where_adv)/tf.reduce_sum(adversarial_ims)
    # optimizers
    #z_opt = tf.train.AdamOptimizer(0.1)
    z_opt = tf.train.MomentumOptimizer(10,momentum=0.9)
    z_minner = z_opt.minimize(total_loss, var_list=z_vars)
    #lam_opt = tf.train.AdamOptimizer(lam_lr)
    lam_opt = tf.train.MomentumOptimizer(10,momentum=0.9)
    lam_maxer = lam_opt.minimize(-total_loss, var_list=[lam])
    with tf.control_dependencies([lam_maxer]):
        lam_update = tf.assign(lam, tf.maximum(0.0, lam)) 
    return total_loss, adv_loss, avg_l2, pct_adv, z_minner, lam_update

def latent_space_attack(eps, overpowered=True, initial_im=None):
    lam_lr = 1.0 #CONFIG["lambda_lr"]
    lam_init = 250.0 #CONFIG["lambda_init"]
    initial_z1 = GAN_MU + GAN_VARIANCE*np.random.randn(BATCH_SIZE,Z_DIM).astype(np.float32)
    initial_z2 = GAN_MU + GAN_VARIANCE*np.random.randn(BATCH_SIZE,Z_DIM).astype(np.float32)
    # latent variable
    z1 = tf.Variable(initial_z1)
    z2 = tf.Variable(initial_z1)
    # Interlude: pre-optimize z1 if we are not overpowered
    z_vars = [z1, z2]
    dep_vars = []
    if not overpowered:
        assert initial_im is not None
        loop, proj_loss = invert(z1, gen_func, \
                            initial_im, \
                            ARGS, \
                            reuse=True,
                            std=GAN_VARIANCE)
        sess.run(remaining_initializer())
        _, proj_loss_ = sess.run([loop, proj_loss])
        log_str = "[log] finished projection step with error %f" % proj_loss_
        print(log_str)
        z1 = sess.run(z1)
        z_vars = [z2]
    # min-max lagrangian variable
    lam = tf.Variable(lam_init)
    # Create graph for attack
    total_loss, adv_loss, avg_l2, pct_adv, z_minner, lam_update = \
        latent_space_graph(z1, z2, lam, eps, z_vars, lam_lr, \
        gen_func, cla_func, TOTAL_SHAPE, L2_EPS)
    # We already initialized classifier and generator
    sess.run(remaining_initializer())
    # Keep track of best attack, not latest
    max_pct = 0.0
    max_adv = None
    if overpowered:
        sess.run(tf.variables_initializer([z1, z2]))
    else:
        sess.run(tf.variables_initializer([z2]))
        sess.run(tf.assign(z2, z1))
    for i in range(ADV_ITERS):
        try:
            loss_, adv_, diff_, pct_, _, _ = sess.run([total_loss, \
                                                adv_loss, \
                                                avg_l2, \
                                                pct_adv, \
                                                z_minner, \
                                                lam_update])
            if pct_ >= max_pct:
                max_pct = pct_
                max_adv = adv_
            if i % 100 == 0:
                print(loss_, adv_, diff_, pct_)
            if EARLY_STOP and (pct_ > ADV_THRESH):
                print("Early stopping at iteration %d, PCT_ADV: %f" % (i,pct_))
                if overpowered:
                    return sess.run([z1, z2])
                else:
                    return z1, sess.run(z2)
        except KeyboardInterrupt as e:
            print("Interrupted by user, outputting logs and quitting...")
            if overpowered:
                return sess.run([z1, z2])
            else:
                return z1, sess.run(z2)
    return None, None

def test_attack(im1, im2, real_labels):
    """ Run all of the above attacks on whatever classifier is specified by the
    params (vanilla, INC, adversarially trained) """
    orig_image = im1.copy()
    if ATTACK_INC:
        z = tf.Variable(GAN_MU + GAN_VARIANCE*np.random.randn(BATCH_SIZE, Z_DIM).astype(np.float32)) 
        x = tf.placeholder(tf.float32, im1.shape)
        loop, proj_loss = invert(z, gen_func, \
                                    x, \
                                    ARGS, \
                                    reuse=True, \
                                    std=GAN_VARIANCE)
        with tf.control_dependencies([loop, proj_loss]):
            gen_img, _ = gen_func(z, reuse=True)
        sess.run(tf.variables_initializer([z]))
        sess.run(tf.variables_initializer([var for var in tf.global_variables() \
            if any([temp in var.name for temp in ["Adam", "Momentum", "beta","RMS"]])]))
        im1 = sess.run(gen_img, {x: im1})
        sess.run(tf.variables_initializer([z]))
        im2 = sess.run(gen_img, {x: im2})
    pl = tf.placeholder(tf.float32, im1.shape)
    logits, preds = cla_func(pl, reuse=True)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, real_labels), tf.float32))
    acc1, acc2 = sess.run(acc, {pl: im1}), sess.run(acc, {pl: im2})
    if (not args.save_dir == "") :
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        for i in range(BATCH_SIZE):
            imsave("%s/adv_%d.png" % (args.save_dir,i), im2[i].reshape(IMG_SHAPE).squeeze())
            if ATTACK_INC:
                imsave("%s/inc_%d.png" % (args.save_dir,i), im1[i].reshape(IMG_SHAPE).squeeze())
            imsave("%s/orig_%d.png" % (args.save_dir,i), orig_image[i].reshape(IMG_SHAPE).squeeze())
    print("Original accuracy: %f | Adversarial accuracy: %f" % (acc1, acc2))

if __name__ == "__main__":
    attack_dict = {}
    z_ = (GAN_MU + np.random.randn(BATCH_SIZE,Z_DIM)*GAN_VARIANCE)
    images, labels = inp_func.inputs(True, DATA_DIR, BATCH_SIZE)
    tf.train.start_queue_runners(sess)
    im, labels_ = sess.run([images, labels])
    if ATTACK_TYPE == "latent_op":
        old_z, new_z = latent_space_attack(EPS, True)
        if new_z is None and old_z is None:
            print("Sorry, attack did not converge!")
        old_im, new_im = sess.run(out, {x: old_z}), sess.run(out, {x: new_z})
    elif ATTACK_TYPE == "latent_normal":
        old_z, new_z = latent_space_attack(EPS, False, im)
        old_im, new_im = sess.run(out, {x: old_z}), sess.run(out, {x: new_z})
    elif ATTACK_TYPE == "fgsm":
        old_im, new_im = fgsm(EPS, im, labels_)
    elif ATTACK_TYPE == "linf_pgd":
        old_im, new_im = iterated_attack(EPS, im, labels_, attack="ifgsm")
    elif ATTACK_TYPE == "pgd":
        old_im, new_im = iterated_attack(EPS, im, labels_, attack="pgd")
    elif ATTACK_TYPE == "l2_pgd":
        old_im, new_im = l2_pgd(EPS, im, labels_)
    elif ATTACK_TYPE == "cw":
        old_im, new_im = cw(im, labels_)
    elif ATTACK_TYPE == "bpda":
        old_im, new_im = bpda(EPS, im, labels_)
    elif ATTACK_TYPE == "clean":
        old_im, new_im = im, im
    test_attack(old_im, new_im, labels_)
