#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: transformation.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np

def gen_pose_shift(bsize, n_pose):
    return np.random.randint(low=-2, high=3, size=[bsize, n_pose], dtype='l')

def gen_affine_trans(bsize):
    s_x = np.random.uniform(low=0.7, high=1.3, size=[bsize, 1])
    s_y = np.random.uniform(low=0.7, high=1.3, size=[bsize, 1])

    t_x = np.random.uniform(low=-0.5, high=0.5, size=[bsize, 1])
    t_y = np.random.uniform(low=-0.5, high=0.5, size=[bsize, 1])
    # t_x = np.random.randint(low=-2, high=3, size=[bsize, 1], dtype='l')
    # t_y = np.random.randint(low=-2, high=3, size=[bsize, 1], dtype='l')

    theta = np.pi * np.random.uniform(low=-0.25, high=0.25, size=[bsize, 1])

    # s_x = np.ones((bsize, 1))*0.7
    # s_y = np.ones((bsize, 1))
    # t_x = np.ones((bsize, 1)) 
    # t_y = np.zeros((bsize, 1))
    # theta = np.pi * np.ones((bsize, 1)) * 0.25

    matrix = np.concatenate(
      (s_x * np.cos(theta), -s_y * np.sin(theta),
      t_x * s_x * np.cos(theta) - t_y * s_y * np.sin(theta),
      s_x * np.sin(theta), s_y * np.cos(theta),
      t_x * s_x * np.sin(theta) + t_y * s_y * np.cos(theta),
      np.zeros((bsize, 1)), np.zeros((bsize, 1)), np.ones((bsize, 1))),
      axis=-1)
    # matrix = np.concatenate(
    #     (s_x * np.cos(theta), -s_y * np.sin(theta),
    #     s_x * np.cos(theta) - s_y * np.sin(theta) + t_x,
    #     s_x * np.sin(theta), s_y * np.cos(theta),
    #     s_x * np.sin(theta) + s_y * np.cos(theta) + t_y,
    #     np.zeros((bsize, 1)), np.zeros((bsize, 1)), np.ones((bsize, 1))),
    #     axis=-1)
    return matrix

if __name__ == '__main__':
    a = gen_affine_trans(5)
    print(np.reshape(a, (-1, 3,3)))
