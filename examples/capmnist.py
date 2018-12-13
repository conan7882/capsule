#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: capmnist.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import sys
import time
import platform
import argparse
import numpy as np
import tensorflow as tf

sys.path.append('../')

from src.nets.capsnet_mnist import CapsNetMNIST, CapsNetMNISTAE
import loader as loader

SAVE_PATH = '/home/qge2/workspace/data/out/capsule/capnet/'

if platform.node() == 'arostitan':
    # data_path = '/home/qge2/workspace/data/homology/'
    MNIST_PATH = '/home/qge2/workspace/data/MNIST_data/'
elif platform.node() == 'Qians-MacBook-Pro.local':
    MNIST_PATH = '/Users/gq/workspace/Dataset/MNIST_data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Train the model.')
    parser.add_argument('--ae', action='store_true',
                        help='add reconstruction')
    parser.add_argument('--eval', action='store_true',
                        help='eval')

    parser.add_argument('--folder', type=str, default='test',
                        help='save folder name')
    parser.add_argument('--load', type=int, default=99,
                        help='Load step of pre-trained')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Init learning rate')
    parser.add_argument('--bsize', type=int, default=128,
                        help='Init learning rate')
    parser.add_argument('--maxepoch', type=int, default=50,
                        help='Max iteration')

    parser.add_argument('--routing', type=int, default=3,
                        help='number of routing')

    return parser.parse_args()

def eval():
    FLAGS = get_args()
    bsize = FLAGS.bsize

    # train_data, test_data = loader.load_mnist(bsize, data_path=MNIST_PATH)
    im_size = 40
    n_channels = 1
    train_data, test_data = loader.two_digits_mnist(canvas_size=im_size, batch_size=bsize)

    n_routing = 3

    if FLAGS.ae:
        CapsNetModel = CapsNetMNISTAE
        model_type = 'ae'
    else:
        CapsNetModel = CapsNetMNIST
        model_type = 'capnet'

    save_path = os.path.join(SAVE_PATH, model_type, FLAGS.folder)
    save_path = save_path + '/'

    test_model = CapsNetModel(
        bsize, im_size, n_channels,
        n_class=10, n_routing=n_routing, n_pred_class=2, translate_im_size=im_size)
    test_model.create_eval_model()

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        writer = tf.summary.FileWriter(save_path)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '{}capnet-epoch-{}'.format(save_path, FLAGS.load))
        test_model.eval_epoch(sess, test_data)

def train():
    FLAGS = get_args()

    train_data, test_data = loader.load_mnist(FLAGS.bsize, data_path=MNIST_PATH)
    im_size = 28
    translate_im_size = 28
    n_channels = 1

    n_routing = 3

    if FLAGS.ae:
        CapsNetModel = CapsNetMNISTAE
        model_type = 'ae'
    else:
        CapsNetModel = CapsNetMNIST
        model_type = 'capnet'

    save_path = os.path.join(SAVE_PATH, model_type, FLAGS.folder)
    save_path = save_path + '/'

    # init training model
    train_model = CapsNetModel(
        FLAGS.bsize, im_size, n_channels, n_class=10, n_routing=n_routing,
        translate_im_size=translate_im_size, shift_range=[[-2, 2], [-2, 2]])
    train_model.create_train_model()

    # init valid model
    valid_model = CapsNetModel(
        FLAGS.bsize, im_size, n_channels, n_class=10, n_routing=n_routing,
        translate_im_size=translate_im_size, shift_range=[[-2, 2], [-2, 2]])
    valid_model.create_valid_model()

    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    
    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        writer = tf.summary.FileWriter(save_path)
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        lr = FLAGS.lr
        for epoch_id in range(FLAGS.maxepoch):
            lr = lr * 0.95
            start_time = time.time()
            train_model.train_epoch(
                sess, train_data, lr, summary_writer=writer)
            print("--- %s seconds ---" % (time.time() - start_time))
            valid_model.valid_epoch(sess, test_data, summary_writer=writer)
            saver.save(sess, '{}capnet-epoch-{}'.format(save_path, epoch_id))
        saver.save(sess, '{}capnet-epoch-{}'.format(save_path, epoch_id))

if __name__ == "__main__":
    FLAGS = get_args()

    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval()



