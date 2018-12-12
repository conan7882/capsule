#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: loader.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import sys
import platform
# import scipy.misc
import scipy.ndimage.interpolation 
import skimage.transform
import numpy as np

sys.path.append('../')
from src.dataflow.mnist import MNISTData, MNISTPair


def load_mnist(batch_size, data_path, shuffle=True, n_use_label=None, n_use_sample=None,
               rescale_size=None):
    """ Function for load training data 

    If n_use_label or n_use_sample is not None, samples will be
    randomly picked to have a balanced number of examples

    Args:
        batch_size (int): batch size
        n_use_label (int): how many labels are used for training
        n_use_sample (int): how many samples are used for training

    Retuns:
        MNISTData dataflow
    """
    # data_path = '/home/qge2/workspace/data/MNIST_data/'

    def preprocess_im(im):
        """ normalize input image to [-1., 1.] """
        if rescale_size is not None:
            im = np.squeeze(im, axis=-1)
            im = skimage.transform.resize(
                im, [rescale_size, rescale_size],
                mode='constant', preserve_range=True)
            im = np.expand_dims(im, axis=-1)
        im = im / 255.

        return np.clip(im, 0., 1.)

    train_data = MNISTData(
        'train',
         data_dir=data_path,
         shuffle=shuffle,
         pf=preprocess_im,
         n_use_label=n_use_label,
         n_use_sample=n_use_sample,
         batch_dict_name=['im', 'label'])
    train_data.setup(epoch_val=0, batch_size=batch_size)

    test_data = MNISTData(
        'test',
         data_dir=data_path,
         shuffle=False,
         pf=preprocess_im,
         n_use_label=n_use_label,
         n_use_sample=n_use_sample,
         batch_dict_name=['im', 'label'])
    test_data.setup(epoch_val=0, batch_size=batch_size)
    return train_data, test_data

def put_two_image(im_1, im_2, im_size, canvas_size):
    pad_h_1 = int(np.floor((canvas_size - im_size) / 2.))
    pad_h_2 = canvas_size - im_size - pad_h_1
    pad_w_1 = int(np.floor((canvas_size - im_size) / 2.))
    pad_w_2 = canvas_size - im_size - pad_w_1

    pad_im_1 = np.pad(
        im_1, ((pad_h_1, pad_h_2), (pad_w_1, pad_w_2)),
        'constant', constant_values=((0, 0), (0, 0)))
    pad_im_2 = np.pad(
        im_2, ((pad_h_1, pad_h_2), (pad_w_1, pad_w_2)),
        'constant', constant_values=((0, 0), (0, 0)))

    off_h = pad_h_1 + 5
    off_w = pad_w_1 + 5
    trans_h = np.random.random(1) * 2. * off_h - off_h
    trans_w = np.random.random(1) * 2. * off_w - off_w

    trans_1 = scipy.ndimage.interpolation.shift(pad_im_1, (trans_h, trans_w))
    trans_2 = scipy.ndimage.interpolation.shift(pad_im_2, (-trans_h, -trans_w))

    return np.maximum(trans_1, trans_2)

def two_digits_mnist(canvas_size, batch_size=128):

    if platform.node() == 'arostitan':
        data_path = '/home/qge2/workspace/data/MNIST_data/'
    elif platform.node() == 'Qians-MacBook-Pro.local':
        data_path = '/Users/gq/workspace/Dataset/MNIST_data/'
    else:
        data_path = 'E:/Dataset/MNIST/'

    def two_digits(im_1, im_2):
        return put_two_image(im_1, im_2, im_size=28, canvas_size=canvas_size)

    label_dict = {}
    label_id = 0
    for i in range(0, 10):
        for j in range(0, 10):
            if '{}{}'.format(i, j) not in label_dict:
                label_dict['{}{}'.format(i, j)] = label_id
                label_dict['{}{}'.format(j, i)] = label_id
                label_id += 1

    train_data = MNISTPair('train',
                           data_dir=data_path,
                           shuffle=True,
                           label_dict=label_dict,
                           batch_dict_name=['im', 'label'],
                           pairprocess=two_digits)
    valid_data = MNISTPair('val',
                           data_dir=data_path,
                           shuffle=True,
                           label_dict=label_dict,
                           batch_dict_name=['im', 'label'],
                           pairprocess=two_digits)

    train_data.setup(epoch_val=0, batch_size=batch_size)
    valid_data.setup(epoch_val=0, batch_size=batch_size)

    return train_data, valid_data

# def load_celeba(batch_size, data_path, rescale_size=64, shuffle=True):
#     """ Load CelebA data

#     Args:
#         batch_size (int): batch size
#         rescale_size (int): rescale image size
#         shuffle (bool): whether shuffle data or not

#     Retuns:
#         CelebA dataflow
#     """
#     # data_path = '/home/qge2/workspace/data/celebA/'
        
#     def face_preprocess(im):
#         offset_h = 50
#         offset_w = 25
#         crop_h = 128
#         crop_w = 128
#         im = im[offset_h: offset_h + crop_h,
#                 offset_w: offset_w + crop_w, :]
#         im = skimage.transform.resize(
#             im, [rescale_size, rescale_size],
#             mode='constant', preserve_range=True)
#         im = im / 255. * 2. - 1.
#         return np.clip(im, -1., 1.)

#     data = CelebA(data_dir=data_path, shuffle=shuffle,
#                   batch_dict_name=['im'], pf_list=[face_preprocess])
#     data.setup(epoch_val=0, batch_size=batch_size)
#     return data

if __name__ == '__main__':
    import matplotlib.pyplot as plt
        
    data, _ = two_digits_mnist(canvas_size=45, batch_size=1)
    batch_data = data.next_batch_dict()

    cur_im = np.squeeze(batch_data['im'][0])
    cur_im = ((cur_im + 1) * 255 / 2)
    cur_im = cur_im.astype(np.uint8)

    print(batch_data['label'][0])

    plt.figure()
    plt.imshow(cur_im)
    plt.show()
