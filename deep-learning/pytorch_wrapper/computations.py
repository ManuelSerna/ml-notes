# Computation-related functions that help with deep learning projects
# ALIAS: comps

import torch
import math


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
    s1 = subsample_shape_out(256, 0, 1, 5, 1)
    s2 = subsample_shape_out(s1, 0, 1, 2, 2)
    s3 = subsample_shape_out(s2, 0, 1, 5, 1)
    s4 = subsample_shape_out(s3, 0, 1, 2, 2)
    print(s1, s2, s3, s4)

