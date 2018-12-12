#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: capsule.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
import src.models.layers as L

SMALL_NUM = 1e-9

def capsule_batch_flatten(x):
    shape = x.get_shape().as_list()[1:-1]
    capsule_size = x.get_shape().as_list()[-1]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape)), capsule_size])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1, capsule_size]))

def squash(inputs, name='squash'):
    # inputs [batch, h, w, dim]
    with tf.name_scope(name):
        s = inputs
        s_norm_2 = tf.reduce_sum(inputs ** 2, axis=-1, keepdims=True)
        s_norm = tf.sqrt(s_norm_2 + SMALL_NUM)
        v = s_norm_2 / (1 + s_norm_2) * s / s_norm
        return v

def shift(pose, pose_transform):
    return pose + pose_transform

def deform(pose, pose_transform):
    return tf.matmul(pose, pose_transform)

def reconstruct_capsule(inputs, num_recogition, num_generation, num_pose, pose_shift,
                        transform_type,
                        wd=0, init_w=None, init_b=tf.zeros_initializer(), bn=False,
                        is_training=True, name='reconstruct_capsule'):
    """
        Capusule use in 'Transforming Auto-encoders'
    """
    with tf.variable_scope(name):
        layer_dict = {'cur_input': inputs}
        recognition = L.linear(
            out_dim=num_recogition, inputs=inputs,               
            init_w=init_w, init_b=init_b, wd=wd, bn=bn,
            is_training=is_training, name='recognition', nl=tf.nn.relu)
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
            is_training=is_training, name='generation', nl=tf.nn.relu)

        input_shape = inputs.get_shape().as_list()
        input_dim = input_shape[1] * input_shape[2] * input_shape[3]
        out = L.linear(
            out_dim=input_dim, inputs=generation,               
            init_w=init_w, init_b=init_b, wd=wd, bn=bn,
            is_training=is_training, name='out', nl=tf.identity)

        out = tf.multiply(out, visual_prob)
        out = tf.reshape(out, shape=[-1, input_shape[1], input_shape[2], input_shape[3]])
        return out

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
    with tf.variable_scope(name):
        layer_dict = {}
        flatten_cap = capsule_batch_flatten(inputs)
        flatten_cap = tf.expand_dims(flatten_cap, axis=2) # [bsize, in_n_cap, 1, in_cap_size]

        # pad_flatten_cap = tf.pad(
        #     flatten_cap,
        #     [[0, 0], [pad_size_1, pad_size_1], [pad_size_2, pad_size_2], [0, 0]],
        #     "REFLECT")

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
        # pred_vec = tf.matmul(flatten_cap_tile, wij, name='pred_vec')
        pred_vec = tf.squeeze(pred_vec, axis=3)
        # pred_vec = L.conv(
        #     filter_size=[in_n_cap, out_n_cap], out_dim=out_cap_size, stride=1,
        #     layer_dict=layer_dict, inputs=flatten_cap_tile, bn=False,
        #     nl=tf.identity, init_w=init_w, use_bias=False, wd=wd,
        #     padding='SAME', trainable=True, is_training=is_training,
        #     name='pred_vec')

        # for cap_id in range(out_n_cap):
        #     uji = L.conv(
        #         filter_size=[in_n_cap, 1], out_dim=out_cap_size, stride=1,
        #         layer_dict=layer_dict, inputs=flatten_cap, bn=False,
        #         nl=tf.identity, init_w=init_w, use_bias=False, wd=wd,
        #         padding='SAME', trainable=True, is_training=is_training,
        #         name='pred_vec_to_cap_{}'.format(cap_id))
        #     pred_vec_list.append(uji)
        # pred_vec = tf.stack(pred_vec_list, axis=2)
        # pred_vec = tf.squeeze(pred_vec, axis=3) # uji matrix [bsize, in_n_cap, out_n_cap, out_cap_size]
        # print(pred_vec)
        
        # in_n_cap = pred_vec.get_shape().as_list()[1]
        # bij = tf.get_variable(
        #     name='rounting_logits', shape=[bsize, in_n_cap, out_n_cap],
        #     initializer=tf.zeros_initializer(), trainable=False)
        # cij = tf.nn.softmax(bij, axis=-1) # [bsize, in_n_cap, out_n_cap]
        # cij = tf.reshape(cij, shape=(bsize, in_n_cap, out_n_cap, 1)) # [bsize, in_n_cap, out_n_cap, 1]
        # sj = tf.reduce_sum(cij * pred_vec, axis=1) # [bsize, out_n_cap, out_cap_size]
        # vj = squash(sj) # [bsize, out_n_cap, out_cap_size]

        # # agreement
        # aij = tf.reduce_sum(pred_vec * tf.expand_dims(vj, axis=1), axis=-1)

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

        return vj, bij, aij




