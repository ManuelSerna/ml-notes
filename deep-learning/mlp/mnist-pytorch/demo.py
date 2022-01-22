import dataset as ds
import gpu_helper_funcs as gf
import mlp
import plotting as pl
import train_helper_funcs as train


import torch
import torchvision

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


if __name__ == '__main__':
    # Get dataset from directory
    dataset = ds.get_mnist(root_dir='data', download=False)
    test_size = 5000
    train_size = len(dataset) - test_size

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

    pl.show_all_batch_imgs(dl=train_loader)

    # Get MLP model
    mnist_model = mlp.MLP(n_features=28*28, n_outputs=10)

    # Switch to device, and move model and data to device
    device = gf.get_default_device()
    train_loader = gf.DeviceDataLoader(train_loader, device)
    test_loader = gf.DeviceDataLoader(test_loader, device)
    mnist_model = gf.move_to_device(mnist_model, device)

    # Train MLP model
    history = train.train_model(
        model=mnist_model,
        epochs=n_epochs,
        lr=0.01,
        train_loader=train_loader,
        val_loader=test_loader,
        opt_func=torch.optim.SGD
    )

    # Visualize loss and accuracy history
    pl.plot_accuracy(history)

    print('Demo finished.')