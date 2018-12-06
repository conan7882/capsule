#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: transform_ae.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf

from src.nets.base import BaseModel
import src.models.transformation as transformation
import src.models.layers as L
import src.models.capsule as capsule_module
import src.utils.viz as viz
from src.models.transformer import spatial_transformer
# import src.models.modules as modules
# import src.models.losses as losses


INIT_W = tf.random_normal_initializer(stddev=0.02)
# INIT_W = tf.keras.initializers.he_normal()

class TransformAE(BaseModel):
    """ class for transforming autoencoder """
    def __init__(self, im_size, n_channels, n_capsule, n_recogition, n_generation, n_pose,
                 transform_type):
        """
        Args:
            im_size (int or list with length 2): size of generate image 
            n_channels (int): number of image channels
        """
        im_size = L.get_shape2D(im_size)
        self.im_h, self.im_w = im_size
        self.n_channels = n_channels
        self.n_capsule = n_capsule
        self.n_recogition = n_recogition
        self.n_generation = n_generation
        self.n_pose = n_pose
        self.transform_type = transform_type

        self.layers = {}

    def _create_train_input(self):
        """ input for training """
        self.pose_shift = tf.placeholder(tf.float32, [None, self.n_pose], name='pose_shift')
        self.image = tf.placeholder(
            tf.float32, [None, self.im_h, self.im_w, self.n_channels],
            name='image')
        if self.transform_type == 'shift':
            self.label = tf.contrib.image.translate(
                self.image,
                translations=self.pose_shift,
                interpolation='NEAREST',
                name='label')
        elif self.transform_type == 'affine':
            T = self.pose_shift[:, :-3]
            T = tf.reshape(T, (-1, 2, 3))
            self.label = spatial_transformer(self.image, T, out_dim=None)

        self.lr = tf.placeholder(tf.float32, name='lr')

    def create_train_model(self):
        """ create graph for training """
        self.set_is_training(True)
        self._create_train_input()
        self.layers['pred'] = self._creat_model(self.image, self.pose_shift)

        self.train_op = self.get_train_op()
        self.loss_op = self.get_loss()
        self.train_summary_op = self.get_train_summary()
        self.global_step = 0
        self.epoch_id = 0

    def  _create_valid_input(self):
        self.pose_shift = tf.placeholder(tf.float32, [None, self.n_pose], name='pose_shift')
        self.image = tf.placeholder(
            tf.float32, [None, self.im_h, self.im_w, self.n_channels],
            name='image')

        if self.transform_type == 'shift':
            self.label = tf.contrib.image.translate(
                self.image,
                translations=self.pose_shift,
                interpolation='NEAREST',
                name='label')
        elif self.transform_type == 'affine':
            T = self.pose_shift[:, :-3]
            T = tf.reshape(T, (-1, 2, 3))
            self.label = spatial_transformer(self.image, T, out_dim=None)

    def create_valid_model(self):
        """ create graph for validation """
        self.set_is_training(False)
        self._create_valid_input()
        self.layers['pred'] = self._creat_model(self.image, self.pose_shift)

        self.loss_op = self.get_loss()
        self.valid_summary_op = self.get_valid_summary()
        self.epoch_id = 0
        
    def _creat_model(self, inputs, pose_shift):
        with tf.variable_scope('transforming_AE', reuse=tf.AUTO_REUSE):
            cap_out = []
            for capsule_id in range(self.n_capsule):
                out = capsule_module.reconstruct_capsule(
                    inputs=inputs, num_recogition=self.n_recogition,
                    num_generation=self.n_generation,
                    num_pose=self.n_pose, pose_shift=pose_shift,
                    wd=0, init_w=INIT_W, bn=False,
                    transform_type=self.transform_type,
                    is_training=self.is_training, name='capsule_{}'.format(capsule_id))
                cap_out.append(out)
            cap_out = tf.add_n(cap_out)

            return tf.nn.sigmoid(cap_out)

    def _get_loss(self):
        label = self.label
        prediction = self.layers['pred']
        loss = tf.reduce_sum((label - prediction) ** 2, axis=[1,2,3])
        return tf.reduce_mean(loss)

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(self.lr)

    def get_train_summary(self):
        with tf.name_scope('train'):
            tf.summary.image('input', self.image, collections=['train'])
            tf.summary.image('label', self.label, collections=['train'])
            tf.summary.image('pred', self.layers['pred'], collections=['train'])
        
        return tf.summary.merge_all(key='train')

    def get_valid_summary(self):
        with tf.name_scope('valid'):
            tf.summary.image('input', self.image, collections=['valid'])
            tf.summary.image('label', self.label, collections=['valid'])
            tf.summary.image('pred', self.layers['pred'], collections=['valid'])
        
        return tf.summary.merge_all(key='valid')

    def train_epoch(self, sess, train_data, init_lr,
                    summary_writer=None):
        """ Train for one epoch of training data

        Args:
            sess (tf.Session): tensorflow session
            train_data (DataFlow): DataFlow for training set
            summary_writer (tf.FileWriter): write for summary. No summary will be
            saved if None.
        """

        display_name_list = ['loss']
        cur_summary = None

        lr = init_lr

        cur_epoch = train_data.epochs_completed
        step = 0
        loss_sum = 0
        self.epoch_id += 1
        while cur_epoch == train_data.epochs_completed:
            self.global_step += 1
            step += 1

            batch_data = train_data.next_batch_dict()
            im = batch_data['im']

            bsize = im.shape[0]
            if self.transform_type == 'shift':
                pose_shift = transformation.gen_pose_shift(bsize, self.n_pose)
            elif self.transform_type == 'affine':
                pose_shift = transformation.gen_affine_trans(bsize)
                
            _, loss = sess.run(
                    [self.train_op, self.loss_op], 
                    feed_dict={self.image: im,
                               self.pose_shift: pose_shift,
                               self.lr: lr})
            loss_sum += loss

            if step % 100 == 0:
                cur_summary = sess.run(
                    self.train_summary_op, 
                    feed_dict={self.image: im,
                               self.pose_shift: pose_shift})

                viz.display(
                    self.global_step,
                    step,
                    [loss_sum],
                    display_name_list,
                    'train',
                    summary_val=cur_summary,
                    summary_writer=summary_writer)

        print('==== epoch: {}, lr:{} ===='.format(cur_epoch, lr))
        cur_summary = sess.run(
            self.train_summary_op, 
            feed_dict={self.image: im,
                       self.pose_shift: pose_shift})
        viz.display(
            self.global_step,
            step,
            [loss_sum],
            display_name_list,
            'train',
            summary_val=cur_summary,
            summary_writer=summary_writer)

    def valid_epoch(self, sess, valid_data, summary_writer=None):
        """ 

        Args:
            sess (tf.Session): tensorflow session
            train_data (DataFlow): DataFlow for training set
            summary_writer (tf.FileWriter): write for summary. No summary will be
            saved if None.
        """

        display_name_list = ['loss']
        cur_summary = None

        valid_data.reset_epochs_completed()
        step = 0
        loss_sum = 0
        self.epoch_id += 1
        while valid_data.epochs_completed < 1:
            step += 1

            batch_data = valid_data.next_batch_dict()
            im = batch_data['im']

            bsize = im.shape[0]
            if self.transform_type == 'shift':
                pose_shift = transformation.gen_pose_shift(bsize, self.n_pose)
            elif self.transform_type == 'affine':
                pose_shift = transformation.gen_affine_trans(bsize)

            loss = sess.run(
                    self.loss_op, 
                    feed_dict={self.image: im,
                               self.pose_shift: pose_shift})
            loss_sum += loss

        print('==== [Valid] ====', end='')
        cur_summary = sess.run(
            self.valid_summary_op, 
            feed_dict={self.image: im,
                       self.pose_shift: pose_shift})
        viz.display(
            self.epoch_id,
            step,
            [loss_sum],
            display_name_list,
            'valid',
            summary_val=cur_summary,
            summary_writer=summary_writer)
