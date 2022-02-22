# Demo
import pytorch_wrapper.display as disp
import pytorch_wrapper.gpu_funcs as gpu_funcs
import pytorch_wrapper.train_eval as te
import pytorch_wrapper.plotting as plot

import vanilla_generator#.MNISTGenerator as MNISTGenerator
import vanilla_discriminator#.MNISTDiscriminator as MNISTDiscriminator

import torch

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


if __name__ == '__main__':
    print('GAN Demo.')

    # Get dataset from directory
    dataset = MNIST(root='data', download=True, train=True, transform=transforms.ToTensor())
    test_size = 5000
    train_size = len(dataset) - test_size

    # TODO: apply transforms to MNIST images (especially normalizing)

    # Create subset datasets for training and testing
    train_subds, test_subds = random_split(
        dataset=dataset,
        lengths=[train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Get DataLoader objects (without worrying about device yet)
    batch_size = 128
    n_epochs = 10

    train_loader = DataLoader(
        dataset=train_subds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_subds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True
    )

    # TODO: Get models


    # TODO: Call training procedure
    #    TODO: create training procedure for vanilla gan
    # and it should output generated images to some directory

    print('Demo finished.')
