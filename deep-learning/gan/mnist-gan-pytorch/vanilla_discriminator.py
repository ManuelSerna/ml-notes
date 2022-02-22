# Vanilla Discriminator for MNIST images
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 28*28
        self.network = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ''' Forward propagate

        :param x: batch of images (as tensors), images are flattened such that (batch_size, 784)
        :return: predictions in range [0, 1]
        '''
        x = x.view(-1, 784)
        self.network(x)
