#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: transformae.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import sys
import platform
import argparse
import numpy as np
import tensorflow as tf

sys.path.append('../')
import loader as loader
from src.nets.transform_ae import TransformAE

SAVE_PATH = '/home/qge2/workspace/data/out/capsule/transform_ae/'
MNIST_PATH = '/home/qge2/workspace/data/MNIST_data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
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

def train():
    FLAGS = get_args()
    save_path = SAVE_PATH

    transform_type = FLAGS.transform

    if FLAGS.transform == 'shift':
        n_capsule = 30
        n_recogition = 10
        n_generation = 20
        n_pose = 2
    elif FLAGS.transform == 'affine':
        n_capsule = 100
        n_recogition = 50
        n_generation = 50
        n_pose = 9

    train_data, test_data = loader.load_mnist(FLAGS.bsize, data_path=MNIST_PATH)
    im_size = 28
    n_channels = 1

    # init training model
    train_model = TransformAE(
        im_size, n_channels, n_capsule, n_recogition, n_generation, n_pose, transform_type)
    train_model.create_train_model()

    # init valid model
    valid_model = TransformAE(
        im_size, n_channels, n_capsule, n_recogition, n_generation, n_pose, transform_type)
    valid_model.create_valid_model()

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
            train_model.train_epoch(
                sess, train_data, lr, summary_writer=writer)
            valid_model.valid_epoch(sess, test_data, summary_writer=writer)
            saver.save(sess, '{}transformae-epoch-{}'.format(save_path, epoch_id))
        saver.save(sess, '{}transformae-epoch-{}'.format(save_path, epoch_id))

if __name__ == "__main__":
    FLAGS = get_args()

    if FLAGS.train:
        train()



