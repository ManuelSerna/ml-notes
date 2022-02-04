# LeNet 5 model in Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        '''
        NOTE: average pooling does not include trainable parameters,
        or sigmoid after average pooling
        '''
        super().__init__()

        # input shape: 1x32x32 (channels, height, width)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        ) # output shape: 6x28x28

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        ) # output shape: 16x5x5

        self.classifier = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
            nn.Softmax(dim=1)
        )

        self.network = nn.Sequential(
            self.conv_block1,
            self.conv_block2,
            nn.Flatten(),
            self.classifier
        )

    def forward(self, x):
        ''' Forward propagate

        :param x: tensor with shape (b, 1, 32, 32) where b is batch size
        :return: prediction on batch of b samples
        '''
        return self.network(x)

if __name__ == '__main__':
    print('[NOTICE] Testing LeNet5.')

    x = torch.rand(1, 1, 32, 32)
    model = LeNet5()
    yhat = model(x)

    print('[NOTICE] Finished testing LeNet5.')
