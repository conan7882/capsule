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
                        help='Train the model.')
    parser.add_argument('--test', action='store_true',
                        help='test')

    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Dataset used for experiment.')

    parser.add_argument('--load', type=int, default=99,
                        help='Load step of pre-trained')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Init learning rate')
    parser.add_argument('--keep_prob', type=float, default=1.,
                        help='keep_prob')
    parser.add_argument('--bsize', type=int, default=128,
                        help='Init learning rate')
    parser.add_argument('--maxepoch', type=int, default=50,
                        help='Max iteration')

    parser.add_argument('--transform', type=str, default='shift',
                        help='Type of transformation.')

    return parser.parse_args()

def test():
    FLAGS = get_args()

    train_data, test_data = loader.two_digits_mnist(canvas_size=46, batch_size=1)
    im_size = 46
    n_channels = 1

    n_routing = 1

    if FLAGS.ae:
        CapsNetModel = CapsNetMNISTAE
        model_type = 'ae'
    else:
        CapsNetModel = CapsNetMNIST
        model_type = 'capnet'

    save_path = os.path.join(SAVE_PATH, model_type)
    save_path = save_path + '/'

    test_model = CapsNetModel(FLAGS.bsize, im_size, n_channels, n_class=10, n_routing=n_routing, n_pred_class=2)
    test_model.create_valid_model()

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        writer = tf.summary.FileWriter(save_path)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, '{}capnet-epoch-{}'.format(save_path, FLAGS.load))
        test_model.test(sess, test_data)

# def ae():
#     FLAGS = get_args()
#     save_path = SAVE_PATH

#     train_data, test_data = loader.load_mnist(FLAGS.bsize, data_path=MNIST_PATH)
#     im_size = 28
#     n_channels = 1

#     # init training model
#     train_model = CapNetMNISTAE(FLAGS.bsize, im_size, n_channels, n_class=10, n_routing=2)
#     train_model.create_train_model()

#     # init valid model
#     valid_model = CapNetMNISTAE(FLAGS.bsize, im_size, n_channels, n_class=10, n_routing=2)
#     valid_model.create_valid_model()

#     sessconfig = tf.ConfigProto()
#     sessconfig.gpu_options.allow_growth = True
#     with tf.Session(config=sessconfig) as sess:
#         writer = tf.summary.FileWriter(save_path)
#         saver = tf.train.Saver()
#         sess.run(tf.global_variables_initializer())
#         writer.add_graph(sess.graph)

#         for epoch_id in range(FLAGS.maxepoch):
#             if epoch_id > 50:
#                 lr = FLAGS.lr / 2.
#             elif epoch_id > 100:
#                 lr = FLAGS.lr / 10.
#             else:
#                 lr = FLAGS.lr
#             train_model.train_epoch(
#                 sess, train_data, lr, summary_writer=writer)
#             valid_model.valid_epoch(sess, test_data, summary_writer=writer)

def train():
    FLAGS = get_args()

    train_data, test_data = loader.load_mnist(FLAGS.bsize, data_path=MNIST_PATH)
    im_size = 28
    n_channels = 1

    n_routing = 1

    if FLAGS.ae:
        CapsNetModel = CapsNetMNISTAE
        model_type = 'ae'
    else:
        CapsNetModel = CapsNetMNIST
        model_type = 'capnet'

    save_path = os.path.join(SAVE_PATH, model_type)
    save_path = save_path + '/'

    # init training model
    train_model = CapsNetModel(FLAGS.bsize, im_size, n_channels, n_class=10, n_routing=n_routing, translate_im_size=28, shift_range=None)
    train_model.create_train_model()

    # init valid model
    valid_model = CapsNetModel(FLAGS.bsize, im_size, n_channels, n_class=10, n_routing=n_routing, translate_im_size=28, shift_range=None)
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
            # if epoch_id > 50:
            #     lr = FLAGS.lr / 2.
            # elif epoch_id > 100:
            #     lr = FLAGS.lr / 10.
            # else:
            #     lr = FLAGS.lr
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
    if FLAGS.test:
        test()



