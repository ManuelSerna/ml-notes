# Dataset-related helper functions
# for Pytorch projects

import torchvision


from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def get_mnist(root_dir=None, download=True):
    ''' Get MNIST image array dataset
    More:
    https://pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html

    :param root_dir: (str) path to MNIST data folder
    :param download: (bool) download dataset to local directory?
    :return: MNIST dataset object (see link)
    '''
    dataset = MNIST(root=root_dir, download=download, train=True, transform=transforms.ToTensor())
    return dataset
