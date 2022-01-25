import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels=None, out_channels=None, kernel_size=(3, 3), padding=1):
    ''' Convolutional block

    :param in_channels: inout channels
    :param out_channels: output channels
    :param kernel_size: (tuple) size of kernel matrices
    :param padding: padding for input data
    :return: torch.nn Sequential object
    '''
    block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        nn.Sigmoid(),
        nn.MaxPool2d(kernel_size=(2, 2))
    )
    return block


class CNN(nn.Module):
    ''' Simple CNN architecture with no optimizations'''
    def __init__(self, in_channels=None, n_outputs=None):
        ''' Constructor

        :param n_channels: (int) number of channels in input
        :param n_outputs: (int) number of outputs for last layer
        '''
        super().__init__()

        self.conv1 = conv_block(in_channels=in_channels, out_channels=32) # out shape: 32x14x14
        self.conv2 = conv_block(in_channels=32, out_channels=64) # out shape: 64x7x7
        self.conv3 = conv_block(in_channels=64, out_channels=128) # out shape: 128x3x3
        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128*3*3, out_features=512),
            nn.Sigmoid(),
            nn.Linear(in_features=512, out_features=128),
            nn.Sigmoid(),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        ''' Forward Propagate

        :param x: tensor in shape (batch x channels x height x width)
        :return: prediction on batch
        '''
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fully_connected(out)
        return out
