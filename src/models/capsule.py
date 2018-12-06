#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: capsule.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import src.models.layers as L


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
