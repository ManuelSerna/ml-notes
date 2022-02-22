# Vanilla Generator Network for MNIST images
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTGenerator(nn.Module):
    def __init__(self, in_features=None):
        ''' Constructor

        :param in_features: dimensionality of random noise vector (size)
        '''
        super().__init__()
        self.noise_vec_size = 100
        self.network = nn.Sequential(
            nn.Linear(in_features=self.noise_vec_size, out_features=128),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=128, out_features=256),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=1024, out_features=28*28),
            nn.Tanh()
        )

    def forward(self, x):
        '''

        :param x: batch of random noise vectors
        :return: generated MNIST images reshaped to (batch_size, 1, 28, 28)
        '''
        return self.network(x).view(-1, 1, 28, 28)
