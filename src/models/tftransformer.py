#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tftransformer.py
# Author: Qian Ge <geqian1001@gmail.com>
# reference:
# https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py

import tensorflow as tf

def translate_image(input_im, trans_im_size, o_im_size, shift_range=None, dtype=tf.float32):
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
            min_h, max_h = -offset_h-5, offset_h+5
            min_w, max_w = -offset_w-5, offset_w+5

        bsize = tf.shape(input_im)[0]
        trans_h = tf.random_uniform(
            (bsize, 1), minval=min_h, maxval=max_h, dtype=dtype)
        trans_w = tf.random_uniform(
            (bsize, 1), minval=min_w, maxval=max_w, dtype=dtype)
        translations = tf.cast(tf.concat((trans_h, trans_w), axis=-1), tf.float32)
        trans_im = tf.contrib.image.translate(
            pad_im, translations,
            interpolation='NEAREST',
            name=None)
        trans_im = tf.reshape(trans_im, (-1, trans_im_size[0], trans_im_size[1], input_im.shape.as_list()[-1]))
        return trans_im, translations


def get_pixel_value(inputs, x, y):
    shape = tf.shape(x)
    bsize = shape[0]

    batch_idx = tf.range(0, bsize)
    batch_idx = tf.reshape(batch_idx, (bsize, 1))
    b = tf.tile(batch_idx, (1, shape[1]))

    indices = tf.stack([b, y, x], axis=-1)
    return tf.gather_nd(inputs, indices)

def spatial_transformer(inputs, T, out_dim=None):
    """
    Args:
        T (tensor): [batch_size, 2, 3]
        outdim (list of 2)

    """
    with tf.name_scope('bilinear_transformer'):
        bsize = tf.shape(inputs)[0]
        channel = tf.shape(inputs)[-1]
        if out_dim is not None:
            h = out_dim
            w = out_dim
            # w = out_dim[1]
        else:
            h = tf.shape(inputs)[1]
            w = tf.shape(inputs)[2]

        o_h = tf.shape(inputs)[1]
        o_w = tf.shape(inputs)[2]

        x = tf.lin_space(-1., 1., w, name='width')
        y = tf.lin_space(-1., 1., h, name='height')
        X, Y = tf.meshgrid(x, y)
        X_flatten = tf.reshape(X, (-1,))
        Y_flatten = tf.reshape(Y, (-1,))
        ones = tf.ones_like(Y_flatten)
        homogeneous_coord = tf.stack((X_flatten, Y_flatten, ones), axis=0) # [3, N]
        homogeneous_coord = tf.expand_dims(homogeneous_coord, axis=0) # [1, 3, N]
        homogeneous_coord = tf.tile(homogeneous_coord, (bsize, 1, 1)) # [bsize, 3, N]
        affine_transform_coord = tf.matmul(T, homogeneous_coord) # [bsize, 2, N]
        affine_x = affine_transform_coord[:, 0, :]
        affine_y = affine_transform_coord[:, 1, :]
        # back to original scale
        affine_x = (affine_x + 1.) / 2. * tf.cast(o_w, tf.float32)
        affine_y = (affine_y + 1.) / 2. * tf.cast(o_h, tf.float32)
        # find four cornors
        x0 = tf.cast(tf.floor(affine_x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(affine_y), tf.int32)
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, 0, o_w - 1)
        x1 = tf.clip_by_value(x1, 0, o_w - 1)
        y0 = tf.clip_by_value(y0, 0, o_h - 1)
        y1 = tf.clip_by_value(y1, 0, o_h - 1)

        # # get value
        I00 = get_pixel_value(inputs, x0, y0)
        I10 = get_pixel_value(inputs, x1, y0)
        I01 = get_pixel_value(inputs, x0, y1)
        I11 = get_pixel_value(inputs, x1, y1)

        x0 = tf.cast(x0, tf.float32)
        x1 = tf.cast(x1, tf.float32)
        y0 = tf.cast(y0, tf.float32)
        y1 = tf.cast(y1, tf.float32)
        # compute weight
        w00 = (y1 - affine_y) * (x1 - affine_x)
        w10 = (y1 - affine_y) * (affine_x - x0)
        w01 = (affine_y - y0) * (x1 - affine_x)
        w11 = (affine_y - y0) * (affine_x - x0)

        w00 = tf.expand_dims(w00, axis=-1)
        w10 = tf.expand_dims(w10, axis=-1)
        w01 = tf.expand_dims(w01, axis=-1)
        w11 = tf.expand_dims(w11, axis=-1)

        transform_im = I00 * w00 + I10 * w10 + I01 * w01 + I11 * w11
        transform_im = tf.reshape(transform_im, (-1, h, w, channel))

        return transform_im

if __name__ == "__main__":
    import os
    import numpy as np
    import imageio
    import matplotlib.pyplot as plt
    from transformation import gen_affine_trans 

    data_path = '/Users/gq/workspace/GitHub/VGG-tensorflow/data/'
    im = imageio.imread(os.path.join(data_path, '000000001584.jpg'))
    # im = np.expand_dims(im, axis=0)
    im = np.stack((im, im), axis=0)
    print(im.shape)

    im_plh = tf.placeholder(tf.float32, [None, None, None, 3])
    im_h = tf.shape(im_plh)[1]
    im_w = tf.shape(im_plh)[2]

    T_plh = tf.placeholder(tf.float32, [None, 2, 3])
    T = [[[1, 0, 0.5], [0, 1, 0]], [[0.5, 0.2, 0], [0.3, 0.5, 0]]]

    T = gen_affine_trans(2)
    print(T)
    T = T[:, :-3]
    T = np.reshape(T, (-1, 2, 3)).astype(np.float32)
    print(T)

    transform_im = spatial_transformer(im_plh, T, out_dim=None)

    with tf.Session() as sess:
        coord_out = sess.run(transform_im, feed_dict={im_plh: im})
        coord_out = coord_out.astype(np.uint8)


    plt.figure()
    plt.imshow(coord_out[1])
    plt.show()
