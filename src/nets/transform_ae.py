#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: transform_ae.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np
import tensorflow as tf

from src.nets.base import BaseModel
import src.utils.viz as viz
import src.models.layers as L
import src.models.capsule as capsule_module
import src.models.transformation as transformation
from src.models.transformer import spatial_transformer


INIT_W = tf.random_normal_initializer(stddev=0.2)
# INIT_W = tf.keras.initializers.he_normal()

class TransformAE(BaseModel):
    """ class for transforming autoencoder from "Transforming Auto-encoders" """
    def __init__(self, im_size, n_channels, n_capsule,
                 n_recognition, n_generation, n_pose, transform_type):
        """
        Args:
            im_size (int or list with length 2): size of generate image 
            n_channels (int): number of image channels
            n_capsule (int): number of capusules
            n_recognition (int): number of recognition unit in each capsule.
            n_generation (int): number of generation unit in each capsule.
            n_pose (int): number of pose unit in each capsule.
            transform_type (str): type of transformation for pose.
                'shift' - pose shift. 'affine' - affine transformation
        """
        im_size = L.get_shape2D(im_size)
        self.im_h, self.im_w = im_size
        self.n_channels = n_channels
        self.n_capsule = n_capsule
        self.n_recognition = n_recognition
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

        # transform input image
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
        else:
            raise ValueError("Unkonwn transform type: {}".format(self.transform_type))

        self.lr = tf.placeholder(tf.float32, name='lr')

    def create_train_model(self):
        """ create graph for training """
        self.set_is_training(True)
        self._create_train_input()
        self.layers['pred'], _, _, _ = self._create_model(self.image, self.pose_shift)

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

        # transform input image
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
        self.layers['pred'], self.layers['pose'], self.layers['visual_prob'], self.layers['transferred_pose']\
            = self._create_model(self.image, self.pose_shift)

        self.loss_op = self.get_loss()
        self.valid_summary_op = self.get_valid_summary()
        self.epoch_id = 0
        
    def _create_model(self, inputs, pose_shift):
        """ create the transform autoencoder """
        with tf.variable_scope('transforming_AE', reuse=tf.AUTO_REUSE):
            cap_out = []
            pose_list = []
            transferred_pose_list = []
            visual_prob_list = []

            for capsule_id in range(self.n_capsule):
                out, pose, visual_prob, transferred_pose = capsule_module.reconstruct_capsule(
                    inputs=inputs, num_recognition=self.n_recognition,
                    num_generation=self.n_generation,
                    num_pose=self.n_pose, pose_shift=pose_shift,
                    wd=0, init_w=INIT_W, bn=False,
                    transform_type=self.transform_type,
                    is_training=self.is_training, name='capsule_{}'.format(capsule_id))
                cap_out.append(out)
                pose_list.append(pose)
                visual_prob_list.append(visual_prob)
                transferred_pose_list.append(transferred_pose)
            cap_out = tf.add_n(cap_out)

            # input_shape = inputs.get_shape().as_list()
            # input_dim = input_shape[1] * input_shape[2] * input_shape[3]
            # out = L.linear(
            #     out_dim=input_dim, inputs=cap_out,               
            #     init_w=INIT_W, wd=0, bn=False,
            #     is_training=self.is_training, name='out', nl=tf.identity)
            # out = tf.reshape(out, shape=[-1, input_shape[1], input_shape[2], input_shape[3]])
            return tf.nn.sigmoid(cap_out), pose_list, visual_prob_list, transferred_pose_list

    def _get_loss(self):
        """ compute the reconstruction loss """
        with tf.name_scope('reconstruction_loss'):
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
            init_lr (float): learning rate
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

            # generate random transformation
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
            valid_data (DataFlow): DataFlow for validation set
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

            # generate random transformation
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

    def viz_batch_test(self, sess, test_data, n_test, save_path, test_type='pose'):
        """ visualize a batch of test data """
        test_data.setup(epoch_val=0, batch_size=n_test)

        if test_type == 'reconstruct':
            batch_data = test_data.next_batch_dict()
            im = batch_data['im']

            # generate random transformation
            if self.transform_type == 'shift':
                pose_shift = transformation.gen_pose_shift(n_test, self.n_pose)
            elif self.transform_type == 'affine':
                pose_shift = transformation.gen_affine_trans(n_test)

            pred, gt = sess.run(
                    [self.layers['pred'], self.label], 
                    feed_dict={self.image: im,
                               self.pose_shift: pose_shift})
            batch_im = np.concatenate((im, gt, pred), axis=0)
            # batch_im = np.concatenate((batch_im, pred), axis=0)
            viz.viz_batch_im(batch_im, [3, n_test], os.path.join(save_path, 'test.png'),
                gap=0, gap_color=0, shuffle=False)

        elif test_type == 'pose':
            assert self.transform_type == 'shift'

            trans_h = tf.ones([n_test, 1]) 
            trans_w = tf.zeros([n_test, 1]) 
            translations_1 = tf.concat((3. * trans_h, trans_w), axis=-1)
            trans_im_1 = tf.contrib.image.translate(
                self.image, translations_1, interpolation='NEAREST')
            translations_2 = tf.concat((-3. * trans_h, trans_w), axis=-1)
            trans_im_2 = tf.contrib.image.translate(
                self.image, translations_2, interpolation='NEAREST')

            trans_h = np.ones([n_test, 1]) 
            trans_w = np.zeros([n_test, 1]) 
            translations_1 = np.concatenate((3. * trans_h, trans_w), axis=-1)
            translations_2 = np.concatenate((-3. * trans_h, trans_w), axis=-1)

            # print(trans_h, trans_w)
            # print(translations_1)

            batch_data = test_data.next_batch_dict()
            im = batch_data['im']

            pose_shift = transformation.gen_pose_shift(n_test, self.n_pose)

            pose, prob, transferred_pose, pred_o = sess.run(
                [self.layers['pose'], self.layers['visual_prob'], self.layers['transferred_pose'], self.layers['pred']],
                feed_dict={self.image: im, self.pose_shift: translations_1})
            shift_im = sess.run(trans_im_1, feed_dict={self.image: im})
            pose_1, prob_1, pred = sess.run(
                [self.layers['pose'], self.layers['visual_prob'], self.layers['pred']],
                feed_dict={self.image: shift_im, self.pose_shift: np.zeros([n_test, 2])})

            # pose_t, prob_t, pred_t = sess.run(
            #     [self.layers['pose'], self.layers['visual_prob'], self.layers['pred']],
            #     feed_dict={self.image: im, self.pose_shift: [[3,0]]})
            shift_im = sess.run(trans_im_2, feed_dict={self.image: im})
            pose_2, prob_2 = sess.run([self.layers['pose'], self.layers['visual_prob']], feed_dict={self.image: shift_im})

            pose_0_list = [ele[0] for ele in pose[:][0]]
            pose_1_list = [ele[0] for ele in pose_1[:][0]]
            pose_2_list = [ele[0] for ele in pose_2[:][0]]
            # print(pose_0_list, pose_1_list)
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(pose_0_list, pose_1_list, 'o')
            plt.plot(pose_0_list, pose_2_list, 'o')
            plt.axis('equal')
            plt.show()
            # print(np.array(pose))
            # print([np.ndarray.tolist(ele) for ele in pose])
            # print('======================== {} ================='.format(translations_1))
            # print([np.ndarray.tolist(ele) for ele in pose_1])
            # print('---------------')
            # print([np.ndarray.tolist(ele1-ele2) for ele1, ele2 in zip(pose_1, pose)])
            # print(prob)
            # print(prob_1)
            # # print(np.array(pose_2))
            # import imageio
            # for i in range(10):
            #     imageio.imwrite(os.path.join(save_path, '{}_input.png'.format(i)), np.squeeze(im[i]))
            #     imageio.imwrite(os.path.join(save_path, '{}_shift_input.png'.format(i)), np.squeeze(shift_im[i]))
            #     imageio.imwrite(os.path.join(save_path, '{}_shift_re.png'.format(i)), np.squeeze(pred[i]))
            #     imageio.imwrite(os.path.join(save_path, '{}_shift_from_o.png'.format(i)), np.squeeze(pred_o[i]))
            
            
        else:
            raise ValueError("Unkonwn test type: {}".format(test_type))

