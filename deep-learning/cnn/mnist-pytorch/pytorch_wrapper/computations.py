# Computation-related functions that help with deep learning projects

import math


def conv_shape_out(size_in=None, padding=None, dilation=None, kernel_size=None, stride=None):
    ''' Compute output shape after one convolution operation

    :param size_in: size of input
    :param padding: padding added to both sides of input
    :param dilation: spacing between elements in kernel matrix
    :param kernel_size: size of convolving kernel
    :param stride: stride of convolution
    :return: size of output after one convolution operation
    '''
    size_out = math.floor(((size_in + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1)
    return size_out


if __name__ == '__main__':
    size = 28
    for i in range(3):
        print(size)
        size = conv_shape_out(
            size_in=size,
            padding=0,
            dilation=1,
            kernel_size=2,
            stride=2
        )
    print(size)
