# Computation-related functions that help with deep learning projects
# ALIAS: comps

import cv2
import torch
import math
import matplotlib.pyplot as plt
import os


def analyze_img_dataset(dir=None):
    ''' Give information on directory containing images

    :param dir: (string) directory/folder containing images
    :return: NA
    '''
    paths = [os.path.join(dir, name) for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]
    heights = []
    widths = []
    N = len(paths)

    for path in paths:
        img = cv2.imread(path)
        heights += [img.shape[0]]
        widths += [img.shape[1]]

    print('Info: {}'.format(dir))
    print('Total Images: {}'.format(N))
    print('Average height of images: {}'.format(sum(heights) // N))
    print('Average width of images: {}'.format(sum(widths) // N))


def avg_list_tensors(tensor_list: list):
    ''' Compute mean of list of (single-value) tensors

    :param tensor_list: list of tensors
    :return: mean of values of each tensor
    '''
    return torch.stack(tensor_list).mean().item()


def get_accuracy(out=None, actual=None):
    ''' Get accuracy

    :param out: model predictions of shape (batch_size,)
    :param actual: labels in a tensor of shape (batch size,)
    :return:
    '''
    _, preds = torch.max(out, dim=1)
    acc = torch.tensor(torch.sum(preds == actual).item() / len(preds))
    return acc


def subsample_shape_out(size_in=None, padding=None, dilation=None, kernel_size=None, stride=None):
    ''' Compute output shape after one subsampling operation
    (e.g., convolution, max pooling, anything that would reduce dims)

    :param size_in: size of input
    :param padding: padding added to both sides of input
    :param dilation: spacing between elements in kernel matrix
    :param kernel_size: size of convolving kernel
    :param stride: stride of convolution
    :return: size of output after one subsampling operation
    '''
    size_out = math.floor(((size_in + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1)
    return size_out


def upsample_shape_out(size_in=None, padding=None, dilation=None, kernel_size=None, stride=None, out_padding=None):
    ''' Compute output shape after one subsampling operation
    (e.g., convolution, max pooling, anything that would increase dims)

    :param size_in: size of input
    :param padding: padding added to both sides of input
    :param dilation: spacing between elements in kernel matrix
    :param kernel_size: size of convolving kernel
    :param stride: stride of convolution
    :param out_padding: additional length added to each side of output
    
    :return: size of output after one upsampling operation
    '''
    size_out = (size_in-1)*stride - 2*padding + dilation*(kernel_size-1) + out_padding + 1
    return size_out


if __name__=='__main__':
    # Testing functionality
    in_size = 64
    s1 = subsample_shape_out(size_in=in_size, padding=1, dilation=1, kernel_size=3, stride=1)
    u1 = upsample_shape_out(size_in=in_size, padding=0, dilation=1, kernel_size=2, stride=2, out_padding=0)
    print('successive conv: {}...{}'.format(in_size, s1))
    print('successive deconv: {}x2={}...{}'.format(in_size, 2*in_size, u1))

    #analyze_img_dataset(dir='../water_bodies/Images')
