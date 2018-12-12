#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tfprocess.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

def translate_image(input_im, trans_im_size, o_im_size, shift_range=None):
    """ Generate random translate images 

        Args:
            input_im: batch of input images [bsize, h, w, c]
            trans_im_size: size of translated image [h, w]
            o_im_size: original image size [h, w]
            shift_range: range of shift [[min_h, max_h], [min_w, max_w]]
    """
    with tf.name_scope('translate_image'):
        offset_h = int((trans_im_size[0] - o_im_size[0]) / 2)
        offset_w = int((trans_im_size[1] - o_im_size[1]) / 2)

        pad_im = tf.pad(
            input_im,
            paddings=tf.constant(
                [[0, 0], [offset_h, offset_h],
                 [offset_w, offset_w], [0, 0]]),
            mode='CONSTANT',
            name='pad_im',
            constant_values=0
            )
        if shift_range is not None:
            min_h, max_h = shift_range[0]
            min_w, max_w = shift_range[1]
        else:
            min_h, max_h = -offset_h, offset_h
            min_w, max_w = -offset_w, offset_w

        bsize = tf.shape(input_im)[0]
        trans_h = tf.random_uniform(
            (bsize, 1), minval=min_h, maxval=max_h)
        trans_w = tf.random_uniform(
            (bsize, 1), minval=min_w, maxval=max_w)
        translations = tf.concat((trans_h, trans_w), axis=-1)
        trans_im = tf.contrib.image.translate(
            pad_im, translations,
            interpolation='NEAREST',
            name=None)
        trans_im = tf.reshape(trans_im, (-1, trans_im_size[0], trans_im_size[1], input_im.shape.as_list()[-1]))
        return trans_im
