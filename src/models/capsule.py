#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: capsule.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
import src.models.layers as L

SMALL_NUM = 1e-9

def capsule_batch_flatten(x):
    """ Flat capsule grid to capsule vector for fc layer
        
        Args:
            x (tensor): tensor of capsule grid. The dim of shape must be greater than 2.
                The first dim is the batch and the last shape is the size of capsule.
                [bsize, h, w, n_capsule_channel, capsule_size]

        Reture:
            tensor of flated capsule [bsize, h*w*n_capsule_channel, capsule_size]
    """
    shape = x.get_shape().as_list()[1:-1]
    capsule_size = x.get_shape().as_list()[-1]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape)), capsule_size])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1, capsule_size]))

def squash(inputs, name='squash'):
    """ squash nonlinearily (1) in "Dynamic Rounting between Capsules"
        
        Args:
            inputs (tensor): input tensor. Dim of shape must be greater than 2.
                The first dim is the batch and the last shape is the size of capsule.
                [batch, h, w, capsule_size] or [batch, len, capsule_size]

        Return:
            squshed capsules with the same shape as inputs
    """

    with tf.name_scope(name):
        s = inputs
        s_norm_2 = tf.reduce_sum(inputs ** 2, axis=-1, keepdims=True)
        s_norm = tf.sqrt(s_norm_2 + SMALL_NUM)
        v = s_norm_2 / (1 + s_norm_2) * s / s_norm
        return v

def shift(pose, pose_transform):
    """ adjust pose by shifting """
    return pose + pose_transform

def deform(pose, pose_transform):
    """ adjust pose by affine transformation. Both inputs are 3x3 transformation matrix """
    return tf.matmul(pose, pose_transform)

def reconstruct_capsule(inputs, num_recognition, num_generation, num_pose, pose_shift,
                        transform_type,
                        wd=0, init_w=None, init_b=tf.zeros_initializer(), bn=False,
                        is_training=True, name='reconstruct_capsule'):
    """ Capsule use in 'Transforming Auto-encoders'.
        This capsule reconstructs the transformed inputs based on pose_shift.
        
        Args:
            inputs (tensor): input tensor [bsize, ...]
            num_recognition (int): number of recognition unit
            num_pose (int): number of pose unit
            pose_shift (tensor): parameters for adjust pose [bsize, ...]
            transform_type (str): type of transformation for pose.
                'shift' - pose shift. 'affine' - affine transformation

        Returns:
            reconstruction of transformed inputs with the same shape as inputs
    """
    with tf.variable_scope(name):
        layer_dict = {'cur_input': inputs}
        recognition = L.linear(
            out_dim=num_recognition, inputs=inputs,               
            init_w=init_w, init_b=init_b, wd=wd, bn=bn,
            is_training=is_training, name='recognition', nl=tf.nn.sigmoid)
        visual_prob = L.linear(
            out_dim=1, inputs=recognition,               
            init_w=init_w, init_b=tf.ones_initializer(), wd=wd, bn=bn,
            is_training=is_training, name='visual_prob', nl=tf.nn.sigmoid)
        pose = L.linear(
            out_dim=num_pose, inputs=recognition,               
            init_w=init_w, init_b=init_b, wd=wd, bn=bn,
            is_training=is_training, name='pose', nl=tf.identity)

        if transform_type == 'shift':
            transferred_pose = pose + pose_shift
        elif transform_type == 'affine':
            pose = tf.reshape(pose, (-1, 3, 3))
            pose_shift = tf.reshape(pose_shift, (-1, 3, 3))
            transferred_pose = tf.matmul(pose, pose_shift)
            transferred_pose = tf.reshape(transferred_pose, (-1, 9))

        generation = L.linear(
            out_dim=num_generation, inputs=transferred_pose,               
            init_w=init_w, init_b=init_b, wd=wd, bn=bn,
            is_training=is_training, name='generation', nl=tf.nn.sigmoid)
        # generation = tf.multiply(generation, visual_prob)

        input_shape = inputs.get_shape().as_list()
        input_dim = input_shape[1] * input_shape[2] * input_shape[3]
        out = L.linear(
            out_dim=input_dim, inputs=generation,               
            init_w=init_w, init_b=init_b, wd=wd, bn=bn,
            is_training=is_training, name='out', nl=tf.identity)

        out = tf.multiply(out, visual_prob)
        out = tf.reshape(out, shape=[-1, input_shape[1], input_shape[2], input_shape[3]])
        return out, pose, visual_prob, transferred_pose

# def conv_capsule_wo_rounting(inputs, bsize, filter_size, stride, n_cap_channel, out_cap_size, 
#                              init_w=None, init_b=tf.zeros_initializer(), wd=0,
#                              is_training=True, name='conv_capsule_wo_rounting'):
#     with tf.variable_scope(name):
#         layer_dict = {}
#         caps_list = []
#         for cap_c_id in range(n_cap_channel):
#             conv_out = L.conv(
#                 filter_size=filter_size, out_dim=out_cap_size, stride=stride,
#                 layer_dict=layer_dict, inputs=inputs, bn=False,
#                 nl=tf.identity, init_w=init_w, init_b=init_b, wd=wd,
#                 padding='VALID', trainable=True, is_training=is_training,
#                 name='cap_grid_{}'.format(cap_c_id))
#             conv_out = squash(conv_out)
#             caps_list.append(conv_out)
#         return tf.stack(caps_list, axis=1)

def conv_capsule_wo_rounting(inputs, bsize, filter_size, stride, n_cap_channel, out_cap_size, 
                             init_w=None, init_b=tf.zeros_initializer(), wd=0,
                             is_training=True, name='conv_capsule_wo_rounting'):
    """ Convolutional capsule without routing in "dynamic routing between capsules"
        The output is a set of capsule grid. Each output channel shares the conv filter.

        Args:
            inputs (tensor): input tensor [bsize, ...]
            bsize (int): batch size. Used for explicity define the shape of output
            filter_size (int or list with len 2): size of conv filter
            stride (int)
            n_cap_channel (int): number of channels of output capsules
            out_cap_size (int): size (dim) of output capsules

        Returns:
            output capsule tensors with size [bsize, h*w, n_cap_channel, out_cap_size].
            h and w determined by the conv

    """
    with tf.variable_scope(name):
        layer_dict = {}
        conv_out = L.conv(
            filter_size=filter_size, out_dim=out_cap_size * n_cap_channel,
            stride=stride,
            layer_dict=layer_dict, inputs=inputs, bn=False,
            nl=tf.identity, init_w=init_w, init_b=init_b, wd=wd,
            padding='VALID', trainable=True, is_training=is_training,
            name='cap_grid')
        grid_size = conv_out.get_shape().as_list()[1:3]
        grid_size = int(np.prod(grid_size))
        conv_out = tf.reshape(conv_out, (bsize, grid_size, n_cap_channel, out_cap_size))
        conv_out = squash(conv_out) # [bsize, h*w, n_cap_channel, out_cap_size]
        return conv_out

def fc_capsule(inputs, bsize, n_routing, out_n_cap, out_cap_size,
               init_w=None, init_b=tf.zeros_initializer(), wd=0,
               is_training=True, name='fc_capsule'):
    """ fully connected capsule layer with routing used in "dynamic routing between capsules"

        Args:
            inputs (tensor): input tensor [bsize, ...]
            bsize (int): batch size. Used for initializing bij
            n_routing (int): number of routing iteration
            out_n_cap (int): number of output capsules
            out_cap_size (int): size (dim) of output capsules

        Returns:
            tensor of output capsules [bsize, out_n_cap, out_cap_size]
    """ 
    with tf.variable_scope(name):
        layer_dict = {}
        flatten_cap = capsule_batch_flatten(inputs)
        flatten_cap = tf.expand_dims(flatten_cap, axis=2) # [bsize, in_n_cap, 1, in_cap_size]

        pred_vec_list = []
        in_n_cap = flatten_cap.get_shape().as_list()[1]
        in_cap_size = flatten_cap.get_shape().as_list()[-1]
        # [bsize, in_n_cap, out_n_cap, in_cap_size]
        flatten_cap_tile = tf.tile(flatten_cap, [1, 1, out_n_cap, 1])
        # uji matrix [bsize, in_n_cap, out_n_cap, out_cap_size]

        wij = tf.get_variable(
            name='wij',
            shape=[in_n_cap, out_n_cap, in_cap_size, out_cap_size],
            initializer=init_w, trainable=True)
        # [bsize, in_n_cap, out_n_cap, 1, in_cap_size]
        flatten_cap_tile = tf.expand_dims(flatten_cap_tile, axis=3)
        pred_vec = tf.scan(lambda a, x: tf.matmul(x, wij), flatten_cap_tile,
            initializer=tf.zeros((in_n_cap, out_n_cap, 1, out_cap_size)))
        pred_vec = tf.squeeze(pred_vec, axis=3) # [bsize, in_n_cap, out_n_cap, out_cap_size]
        
        in_n_cap = pred_vec.get_shape().as_list()[1]
        bij = tf.zeros(shape=[bsize, in_n_cap, out_n_cap])

        for i in range(n_routing + 1):
            with tf.name_scope('routing_{}'.format(i)):
                cij = tf.nn.softmax(bij, axis=-1) # [bsize, in_n_cap, out_n_cap]
                cij = tf.reshape(cij, shape=(bsize, in_n_cap, out_n_cap, 1)) # [bsize, in_n_cap, out_n_cap, 1]
                sj = tf.reduce_sum(cij * pred_vec, axis=1) # [bsize, out_n_cap, out_cap_size]
                vj = squash(sj) # [bsize, out_n_cap, out_cap_size]

                if i < n_routing:
                    # agreement
                    aij = tf.reduce_sum(pred_vec * tf.expand_dims(vj, axis=1), axis=-1)
                    bij += aij

        return vj, bij




