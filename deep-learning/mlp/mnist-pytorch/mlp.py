import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    ''' Simple Multilayer Perceptron/Neural Network'''
    def __init__(self, n_features=None, n_outputs=None):
        ''' Constructor

        :param n_features: (int) number of input features for first layer
        :param n_outputs: (int) number of outputs for last layer
        '''
        super().__init__()

        self.linear1 = nn.Linear(in_features=n_features, out_features=16)
        self.linear2 = nn.Linear(in_features=16, out_features=16)
        self.out1 = nn.Linear(in_features=16, out_features=n_outputs)

    def forward(self, x=None):
        ''' Forward propagation to get prediction

        :param x: tensor for data
        :return: model prediction
        '''
        out = x.view(x.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.out1(out)
        return out
