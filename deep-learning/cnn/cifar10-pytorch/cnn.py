import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_wrapper.computations as comps


def conv_block(in_channels=None, out_channels=None, kernel_size=(3, 3), padding=1, pooling=False):
    ''' Convolutional block

    :param in_channels: inout channels
    :param out_channels: output channels
    :param kernel_size: (tuple) size of kernel matrices
    :param padding: padding for input data
    :param pooling: (bool) apply max pooling to have img size?
    :return: torch.nn Sequential object
    '''
    ops = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU()
    ]

    if pooling:
        ops.append(nn.MaxPool2d(kernel_size=(2, 2)))

    return nn.Sequential(*ops)


class CIFAR10CNN(nn.Module):
    ''' Simple CNN architecture for CIFAR10 images
    NOTE: it is assumed width=height and are the same for all images'''
    def __init__(self, in_channels=None, n_outputs=None, img_size=32):
        ''' Constructor

        :param in_channels: (int) number of channels in input
        :param n_outputs: (int) number of outputs for last layer
        :param img_size: (int) width/height of image
        '''
        super().__init__()

        # size: 32x32 -> 16x16
        self.conv1 = conv_block(in_channels=in_channels, out_channels=16)
        self.conv2 = conv_block(in_channels=16, out_channels=32, pooling=True)
        out_size = comps.subsample_shape_out(size_in=img_size, padding=0, dilation=1, kernel_size=2, stride=2)

        # size: 16x16 -> 8x8
        self.conv3 = conv_block(in_channels=32, out_channels=64)
        self.conv4 = conv_block(in_channels=64, out_channels=128, pooling=True)
        out_size = comps.subsample_shape_out(size_in=out_size, padding=0, dilation=1, kernel_size=2, stride=2)

        # size: 8x8 -> 4x4
        self.conv5 = conv_block(in_channels=128, out_channels=256)
        self.conv6 = conv_block(in_channels=256, out_channels=256, pooling=True)
        out_size = comps.subsample_shape_out(size_in=out_size, padding=0, dilation=1, kernel_size=2, stride=2)

        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256*out_size*out_size, out_features=512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_outputs)
        )

    def forward(self, x):
        ''' Forward Propagate

        :param x: tensor in shape (batch x channels x height x width)
        :return: prediction on batch
        '''
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.fully_connected(out)
        return out
