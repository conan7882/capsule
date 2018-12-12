#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: capnet_mnist.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf

from src.nets.base import BaseModel
# import src.models.transformation as transformation
import src.models.layers as L
import src.models.capsule as capsule_module
import src.utils.viz as viz
# from src.models.transformer import spatial_transformer
# import src.models.modules as modules
# import src.models.losses as losses


INIT_W = tf.random_normal_initializer(stddev=0.02)
# INIT_W = tf.keras.initializers.he_normal()

class CapNetMNIST(BaseModel):
    """ class for transforming autoencoder """
    def __init__(self, batch_size, im_size, n_channels, n_class, n_routing):
        """
        Args:
            im_size (int or list with length 2): size of generate image 
            n_channels (int): number of image channels
        """
        im_size = L.get_shape2D(im_size)
        self.im_h, self.im_w = im_size
        self.n_channels = n_channels
        self.n_class = n_class
        self.bsize = batch_size
        self.n_routing = n_routing

        self.layers = {}

    def _create_train_input(self):
        """ input for training """
        self.image = tf.placeholder(
            tf.float32, [None, self.im_h, self.im_w, self.n_channels],
            name='image')
        self.label = tf.placeholder(tf.int64, [None], 'label')
        self.lr = tf.placeholder(tf.float32, name='lr')

    def create_train_model(self):
        """ create graph for training """
        self.set_is_training(True)
        self._create_train_input()
        self.layers['DigitCaps'], self.layers['bij'], self.layers['aij'] = self._cap_encoder(self.image)
        self.layers['pred'] = self.get_prediction()

        self.train_op = self.get_train_op()
        self.loss_op = self.get_loss()
        self.routing_op = self.routing()
        self.init_routing_op = self.init_routing()
        self.accuracy_op = self.get_accuracy()
        self.train_summary_op = self.get_train_summary()
        self.global_step = 0
        self.epoch_id = 0

    def  _create_valid_input(self):
        self.image = tf.placeholder(
            tf.float32, [None, self.im_h, self.im_w, self.n_channels],
            name='image')
        self.label = tf.placeholder(tf.int64, [None], 'label')

    def create_valid_model(self):
        """ create graph for validation """
        self.set_is_training(False)
        self._create_valid_input()
        self.layers['DigitCaps'], self.layers['bij'], self.layers['aij'] = self._cap_encoder(self.image)
        self.layers['pred'] = self.get_prediction()

        self.loss_op = self.get_loss()
        self.accuracy_op = self.get_accuracy()
        self.routing_op = self.routing()
        self.init_routing_op = self.init_routing()
        self.valid_summary_op = self.get_valid_summary()
        self.epoch_id = 0
        
    def _cap_encoder(self, inputs):
        with tf.variable_scope('CapNetEncoder', reuse=tf.AUTO_REUSE):
            bij_list = []
            aij_list = []
            conv_out = L.conv(
                filter_size=9, out_dim=256, stride=1,
                layer_dict=self.layers, inputs=inputs, bn=False,
                nl=tf.nn.relu, init_w=INIT_W, wd=0,
                padding='VALID', trainable=True, is_training=self.is_training,
                name='conv_1')

            primary_caps = capsule_module.conv_capsule_wo_rounting(
                inputs=conv_out, filter_size=9, stride=2, n_cap_channel=32, out_cap_size=8, 
                init_w=INIT_W, wd=0, is_training=self.is_training,
                name='conv_capsule_wo_rounting')

            digit_caps, bij, aij = capsule_module.fc_capsule(
                inputs=primary_caps, bsize=self.bsize, out_n_cap=self.n_class, out_cap_size=16,
                init_w=INIT_W, wd=0, is_training=self.is_training,
                name='fc_capsule')
            bij_list.append(bij)
            aij_list.append(aij)

        return digit_caps, bij_list, aij_list

    def get_prediction(self):
        with tf.name_scope('prediction'):
            digit_caps = self.layers['DigitCaps'] # [bsize, n_class, size_cap]
            caps_norm = tf.reduce_sum(digit_caps ** 2, axis=-1) # [bsize, n_class]
            pred = tf.argmax(caps_norm, axis=-1)
            return pred

    def init_routing(self):
        with tf.name_scope('init_routing'):
            op_list = []
            for bij in self.layers['bij']:
                op = bij.assign(tf.zeros_like(bij))
                op_list.append(op)
            return op_list

    def routing(self):
        with tf.name_scope('routing'):
            update_list = []
            for bij, aij in zip(self.layers['bij'], self.layers['aij']):
                new_bij = bij + aij
                update = bij.assign(new_bij)
                update_list.append(update)

        return update_list

    def _get_loss(self):
        with tf.name_scope('loss'):
            margin_loss = self._get_margin_loss()

        return margin_loss

    def _get_margin_loss(self):
        with tf.name_scope('margin_loss'):
            label = self.label # [bsize]
            label_one_hot = tf.one_hot(label, self.n_class) # [bsize, n_class]

            digit_caps = self.layers['DigitCaps'] # [bsize, n_class, size_cap]
            caps_norm = tf.sqrt(tf.reduce_sum(digit_caps ** 2, axis=-1)) # [bsize, n_class]

            down_weight = 0.5
            m_plus = 0.9
            m_minus = 0.1

            loss = label_one_hot * (tf.nn.relu(m_plus - caps_norm) ** 2)\
                + down_weight * (1 - label_one_hot) * (tf.nn.relu(caps_norm - m_minus) ** 2)

            return tf.reduce_mean(loss)

    def get_accuracy(self):
        with tf.name_scope('accuracy'):
            label = self.label
            pred = self.layers['pred']
            acc = tf.reduce_mean(tf.cast(tf.equal(label, pred), tf.float32))
            return acc

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(self.lr)

    def get_train_summary(self):
        with tf.name_scope('train'):
            for i, bij in enumerate(self.layers['bij']):
                tf.summary.histogram('bij_{}'.format(i), bij, collections=['train'])        
        return tf.summary.merge_all(key='train')

    def get_valid_summary(self):
        with tf.name_scope('valid'):
            for i, bij in enumerate(self.layers['bij']):
                tf.summary.histogram('bij_{}'.format(i), bij, collections=['valid'])        
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

        display_name_list = ['loss', 'accuracy']
        cur_summary = None

        lr = init_lr

        cur_epoch = train_data.epochs_completed
        step = 0
        loss_sum = 0
        acc_sum = 0
        self.epoch_id += 1
        while cur_epoch == train_data.epochs_completed:
            self.global_step += 1
            step += 1

            batch_data = train_data.next_batch_dict()
            im = batch_data['im']
            label = batch_data['label']

            # routing
            sess.run(self.init_routing_op)

            for routing_op in self.routing_op:
                for i in range(self.n_routing):
                    sess.run(routing_op, 
                             feed_dict={self.image: im})
                
            _, loss, acc = sess.run(
                    [self.train_op, self.loss_op, self.accuracy_op], 
                    feed_dict={self.image: im,
                               self.label: label,
                               self.lr: lr})

            loss_sum += loss
            acc_sum += acc
            # print(test)

            if step % 100 == 0:
                cur_summary = sess.run(
                    self.train_summary_op, 
                    feed_dict={self.image: im,
                               self.label: label})

                viz.display(
                    self.global_step,
                    step,
                    [loss_sum, acc_sum],
                    display_name_list,
                    'train',
                    summary_val=cur_summary,
                    summary_writer=summary_writer)

        print('==== epoch: {}, lr:{} ===='.format(cur_epoch, lr))
        cur_summary = sess.run(
            self.train_summary_op, 
            feed_dict={self.image: im,
                       self.label: label})
        viz.display(
            self.global_step,
            step,
            [loss_sum, acc_sum],
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

        display_name_list = ['loss', 'accuracy']
        cur_summary = None

        valid_data.reset_epochs_completed()
        step = 0
        loss_sum = 0
        acc_sum = 0
        self.epoch_id += 1
        while valid_data.epochs_completed < 1:
            step += 1

            batch_data = valid_data.next_batch_dict()
            im = batch_data['im']
            label = batch_data['label']

            # routing
            sess.run(self.init_routing_op)

            for routing_op in self.routing_op:
                for i in range(self.n_routing):
                    sess.run(routing_op, 
                             feed_dict={self.image: im})

            loss, acc = sess.run(
                [self.loss_op, self.accuracy_op], 
                feed_dict={self.image: im,
                           self.label: label})
            loss_sum += loss
            acc_sum += acc

        print('==== [Valid] ====', end='')
        cur_summary = sess.run(
            self.valid_summary_op, 
            feed_dict={self.image: im})
        viz.display(
            self.epoch_id,
            step,
            [loss_sum, acc_sum],
            display_name_list,
            'valid',
            summary_val=cur_summary,
            summary_writer=summary_writer)

class CapNetMNISTAE(CapNetMNIST):
    def _cap_decoder(self, inputs):
        # inputs [bsize, n_class, cap_size]
        with tf.variable_scope('CapNetDecoder', reuse=tf.AUTO_REUSE):
            caps_norm = tf.reduce_sum(inputs ** 2, axis=-1) # [bsize, n_class]
            pred = tf.cast(tf.argmax(caps_norm, axis=-1), tf.int32) # [bsize]
            gather_id = tf.constant([i for i in range(self.bsize)])
            gather_id = tf.concat([tf.expand_dims(gather_id, axis=-1), tf.expand_dims(pred, axis=-1)], axis=-1)
            
            masked_cap = tf.gather_nd(inputs, gather_id, name='mask_cap')

            arg_scope = tf.contrib.framework.arg_scope
            with arg_scope([L.linear], 
                           layer_dict=self.layers, bn=False,
                           init_w=INIT_W, wd=0, is_training=self.is_training):
                self.layers['cur_input'] = masked_cap
                L.linear(out_dim=512, name='fc_1', nl=tf.nn.relu)
                L.linear(out_dim=1024, name='fc_2', nl=tf.nn.relu)
                L.linear(out_dim=784, name='output', nl=tf.sigmoid)
            out = tf.reshape(self.layers['cur_input'], (-1, self.im_h, self.im_w, self.n_channels))
        return out

    def create_train_model(self):
        """ create graph for training """
        self.set_is_training(True)
        self._create_train_input()
        self.layers['DigitCaps'], self.layers['bij'], self.layers['aij'] = self._cap_encoder(self.image)
        self.layers['pred'] = self.get_prediction()
        self.layers['reconstruction'] = self._cap_decoder(self.layers['DigitCaps'])

        self.train_op = self.get_train_op()
        self.loss_op = self.get_loss()
        self.routing_op = self.routing()
        self.init_routing_op = self.init_routing()
        self.accuracy_op = self.get_accuracy()
        self.train_summary_op = self.get_train_summary()
        self.global_step = 0
        self.epoch_id = 0

    def create_valid_model(self):
        """ create graph for validation """
        self.set_is_training(False)
        self._create_valid_input()
        self.layers['DigitCaps'], self.layers['bij'], self.layers['aij'] = self._cap_encoder(self.image)
        self.layers['pred'] = self.get_prediction()
        self.layers['reconstruction'] = self._cap_decoder(self.layers['DigitCaps'])

        self.loss_op = self.get_loss()
        self.accuracy_op = self.get_accuracy()
        self.routing_op = self.routing()
        self.init_routing_op = self.init_routing()
        self.valid_summary_op = self.get_valid_summary()
        self.epoch_id = 0

    def _get_reconstruction_loss(self):
        with tf.name_scope('reconstruction_loss'):
            label = self.image
            pred = self.layers['reconstruction']
            loss = tf.reduce_sum((label - pred) ** 2, axis=[1, 2, 3])
            loss = tf.reduce_mean(loss)
            return loss

    def _get_loss(self):
        margin_loss = self._get_margin_loss()
        rect_loss = self._get_reconstruction_loss()
        loss = margin_loss + 0.0005 * rect_loss
        return loss

    def get_train_summary(self):
        with tf.name_scope('train'):
            for i, bij in enumerate(self.layers['bij']):
                tf.summary.histogram('bij_{}'.format(i), bij, collections=['train'])
            tf.summary.image('input', self.image, collections=['train'])
            tf.summary.image('reconstruction', self.layers['reconstruction'], collections=['train'])
        
        return tf.summary.merge_all(key='train')

    def get_valid_summary(self):
        with tf.name_scope('valid'):
            for i, bij in enumerate(self.layers['bij']):
                tf.summary.histogram('bij_{}'.format(i), bij, collections=['valid'])
            tf.summary.image('input', self.image, collections=['valid'])
            tf.summary.image('reconstruction', self.layers['reconstruction'], collections=['valid'])       
        return tf.summary.merge_all(key='valid')