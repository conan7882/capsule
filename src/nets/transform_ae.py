#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: transform_ae.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np
import tensorflow as tf

import src.utils.viz as viz
import src.models.layers as L
from src.nets.base import BaseModel
import src.models.capsule as capsule_module
import src.models.transformation as transformation
from src.models.tftransformer import spatial_transformer, translate_image


INIT_W = tf.random_normal_initializer(stddev=0.002)
# INIT_W = tf.keras.initializers.he_normal()

class TransformAE(BaseModel):
    """ class for transforming autoencoder from "Transforming Auto-encoders" """
    def __init__(self, im_size, n_channels, n_capsule,
                 n_recognition, n_generation, n_pose, transform_type,
                 wd=0, translate_input=False):
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
            translate_input (bool): Weather randomly translate input images or not.
                Used only when transform_type == 'shift'
        """
        im_size = L.get_shape2D(im_size)
        self.im_h, self.im_w = im_size
        self.n_channels = n_channels
        self.n_capsule = n_capsule
        self.n_recognition = n_recognition
        self.n_generation = n_generation
        self.n_pose = n_pose
        self.transform_type = transform_type
        self.translate_input = translate_input
        self.wd = wd

        self.layers = {}

    def _create_train_input(self):
        """ input for training """
        self.raw_pose_shift = tf.placeholder(tf.float32, [None, self.n_pose], name='pose_shift')
        self.raw_image = tf.placeholder(
            tf.float32, [None, self.im_h, self.im_w, self.n_channels],
            name='image')

        if self.transform_type == 'shift' and self.translate_input:
            self.image, raw_shift = translate_image(
                self.raw_image,
                [self.im_h, self.im_w],
                [self.im_h, self.im_h],
                shift_range=[[-2, 3], [-2, 3]],
                dtype=tf.int32)
            self.pose_shift = self.raw_pose_shift - raw_shift
        else:
            self.image = self.raw_image
            self.pose_shift = self.raw_pose_shift
        # transform input image
        if self.transform_type == 'shift':
            self.label = tf.contrib.image.translate(
                self.raw_image,
                translations=self.raw_pose_shift,
                interpolation='NEAREST',
                name='label')
        elif self.transform_type == 'affine':
            T = self.pose_shift[:, :-3]
            T = tf.reshape(T, (-1, 2, 3))
            self.label = spatial_transformer(self.raw_image, T, out_dim=None)
        else:
            raise ValueError("Unkonwn transform type: {}".format(self.transform_type))

        self.lr = tf.placeholder(tf.float32, name='lr')

    def create_train_model(self):
        """ create graph for training """
        self.set_is_training(True)
        self._create_train_input()
        self.layers['pred'], _, _, _, _ = self._create_model(self.image, self.pose_shift)

        self.train_op = self.get_train_op()
        self.loss_op = self.get_loss()
        self.train_summary_op = self.get_train_summary()
        self.global_step = 0
        self.epoch_id = 0

    def  _create_valid_input(self):
        self.raw_pose_shift = tf.placeholder(
            tf.float32, [None, self.n_pose], name='pose_shift')
        self.raw_image = tf.placeholder(
            tf.float32, [None, self.im_h, self.im_w, self.n_channels],
            name='image')

        if self.transform_type == 'shift' and self.translate_input:
            self.image, raw_shift = translate_image(
                self.raw_image,
                [self.im_h, self.im_w],
                [self.im_h, self.im_h],
                shift_range=[[-2, 3], [-2, 3]],
                dtype=tf.int32)
            self.pose_shift = self.raw_pose_shift - raw_shift
        else:
            self.image = self.raw_image
            self.pose_shift = self.raw_pose_shift

        # transform input image
        if self.transform_type == 'shift':
            self.label = tf.contrib.image.translate(
                self.raw_image,
                translations=self.raw_pose_shift,
                interpolation='NEAREST',
                name='label')
        elif self.transform_type == 'affine':
            T = self.pose_shift[:, :-3]
            T = tf.reshape(T, (-1, 2, 3))
            self.label = spatial_transformer(self.raw_image, T, out_dim=None)

    def create_valid_model(self):
        """ create graph for validation """
        self.set_is_training(False)
        self._create_valid_input()
        self.layers['pred'], self.layers['pose'], self.layers['visual_prob'],\
        self.layers['transferred_pose'], self.layers['out_weights']\
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
            out_weights_list = []

            for capsule_id in range(self.n_capsule):
                out, pose, visual_prob, transferred_pose, out_weights = capsule_module.reconstruct_capsule(
                    inputs=inputs, num_recognition=self.n_recognition,
                    num_generation=self.n_generation,
                    num_pose=self.n_pose, pose_shift=pose_shift,
                    wd=self.wd, init_w=INIT_W, bn=False,
                    transform_type=self.transform_type,
                    is_training=self.is_training, name='capsule_{}'.format(capsule_id))
                cap_out.append(out)
                pose_list.append(pose)
                visual_prob_list.append(visual_prob)
                transferred_pose_list.append(transferred_pose)
                out_weights_list.append(out_weights)
            cap_out = tf.add_n(cap_out)

            return tf.nn.sigmoid(cap_out), pose_list, visual_prob_list, transferred_pose_list, out_weights_list

    def _get_loss(self):
        """ compute the reconstruction loss """
        with tf.name_scope('reconstruction_loss'):
            label = self.label
            prediction = self.layers['pred']
            loss = tf.reduce_sum((label - prediction) ** 2, axis=[1,2,3])
            # tf.add_to_collection('losses', loss)
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
                    feed_dict={self.raw_image: im,
                               self.raw_pose_shift: pose_shift,
                               self.lr: lr})
            loss_sum += loss

            if step % 100 == 0:
                cur_summary = sess.run(
                    self.train_summary_op, 
                    feed_dict={self.raw_image: im,
                               self.raw_pose_shift: pose_shift})

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
            feed_dict={self.raw_image: im,
                       self.raw_pose_shift: pose_shift})
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
                    feed_dict={self.raw_image: im,
                               self.raw_pose_shift: pose_shift})
            loss_sum += loss

        print('==== [Valid] ====', end='')
        cur_summary = sess.run(
            self.valid_summary_op, 
            feed_dict={self.raw_image: im,
                       self.raw_pose_shift: pose_shift})
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
        if test_type == 'single_pose':
            n_test = 1
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
                    feed_dict={self.raw_image: im,
                               self.raw_pose_shift: pose_shift})
            batch_im = np.concatenate((im, gt, pred), axis=0)
            # batch_im = np.concatenate((batch_im, pred), axis=0)
            viz.viz_batch_im(
                batch_im, [3, n_test],
                os.path.join(save_path, 'reconstruct_{}.png'.format(self.transform_type)),
                gap=0, gap_color=0, shuffle=False)

        elif test_type == 'filter':
            weights = sess.run(self.layers['out_weights'])
            for i in range(self.n_capsule):
                viz.viz_batch_im(
                    weights[i], [2, 10], os.path.join(save_path, 'filter_{}.png'.format(i)),
                    gap=0, gap_color=0, shuffle=False)

        elif test_type == 'pose' or test_type == 'single_pose':
            import matplotlib.pyplot as plt

            if self.transform_type == 'affine':
                batch_data = test_data.next_batch_dict()
                im = batch_data['im']
                pose, prob = sess.run(
                    [self.layers['pose'], self.layers['visual_prob']],
                    feed_dict={self.raw_image: im})

                mean_pro = np.mean(prob, axis=1)
                fig = plt.figure()
                plt.plot(mean_pro, 'o-')
                plt.xlabel("Capsules")
                plt.ylabel("Probability")
                plt.grid()
                # plt.xlim(-5, self.n_capsule + 1) 
                fig.savefig(os.path.join(save_path, 'prob.png'), dpi=fig.dpi)
            else:
                shift_val = 3.
                trans_h = tf.ones([n_test, 1]) 
                trans_w = tf.zeros([n_test, 1]) 
                translations_1 = tf.concat((shift_val * trans_h, trans_w), axis=-1)
                trans_im_1 = tf.contrib.image.translate(
                    self.raw_image, translations_1, interpolation='NEAREST')
                translations_2 = tf.concat((-shift_val * trans_h, trans_w), axis=-1)
                trans_im_2 = tf.contrib.image.translate(
                    self.raw_image, translations_2, interpolation='NEAREST')

                batch_data = test_data.next_batch_dict()
                im = batch_data['im']
                pose, prob = sess.run(
                    [self.layers['pose'], self.layers['visual_prob']],
                    feed_dict={self.raw_image: im})

                shift_im = sess.run(trans_im_1, feed_dict={self.raw_image: im})
                pose_1, prob_1 = sess.run(
                    [self.layers['pose'], self.layers['visual_prob']],
                    feed_dict={self.raw_image: shift_im})

                shift_im = sess.run(trans_im_2, feed_dict={self.raw_image: im})
                pose_2, prob_2 = sess.run(
                    [self.layers['pose'], self.layers['visual_prob']],
                    feed_dict={self.raw_image: shift_im})

                if test_type == 'single_pose':
                    plt.figure()
                    plt.bar(range(self.n_capsule), prob)
                    diff_1 = [(p_1[0] - p_0[0])[0] for p_0, p_1 in zip(pose, pose_1)]
                    diff_2 = [(p_2[0] - p_0[0])[0] for p_0, p_2 in zip(pose, pose_2)]
                    fig = plt.figure()
                    plt.plot(diff_1, 'o')
                    plt.plot(diff_2, 'o')
                    plt.plot(shift_val * np.ones(len(diff_1)))
                    plt.plot(-shift_val * np.ones(len(diff_1)))
                    plt.show()

                else:
                    mean_pro = np.mean(prob, axis=1)
                    mean_pro_1 = np.mean(prob_1, axis=1)
                    mean_pro_2 = np.mean(prob_2, axis=1)
                    fig = plt.figure()
                    plt.plot(mean_pro, 'o-')
                    plt.plot(mean_pro_1, 'o-')
                    plt.plot(mean_pro_2, 'o-')
                    plt.xlabel("Capsules")
                    plt.ylabel("Probability")
                    plt.grid()
                    plt.xlim(-5, self.n_capsule + 1) 
                    plt.legend(['Original', 'Shift +{} pixel'.format(shift_val), 'Shift -{} pixel'.format(shift_val)], loc='upper left')
                    fig.savefig(os.path.join(save_path, 'prob.png'), dpi=fig.dpi) 

                    for i in range(self.n_capsule):
                        pose_0_list = [ele[0] for ele in pose[i]]
                        pose_1_list = [ele[0] for ele in pose_1[i]]
                        pose_2_list = [ele[0] for ele in pose_2[i]]
                        fig = plt.figure()
                        # ax = Axes3D(fig)
                        # ax.scatter(pose_0_list, pose_1_list, prob_list, c='C1', marker='o')
                        plt.plot(pose_1_list, pose_0_list, 'o', color='C1')
                        plt.plot(pose_2_list, pose_0_list, 'o', color='C2')
                        plt.plot(pose_0_list - shift_val * np.ones(n_test), pose_0_list, color='C2')
                        plt.plot(pose_0_list + shift_val * np.ones(n_test), pose_0_list, color='C1')
                        plt.xlabel("+{}/-{} pixel shift in x direction".format(shift_val, shift_val))
                        plt.ylabel("original x output")
                        plt.axis('equal')
                        fig.savefig(os.path.join(save_path, '{}_x_output.png'.format(i)), dpi=fig.dpi)            
        else:
            raise ValueError("Unkonwn test type: {}".format(test_type))

