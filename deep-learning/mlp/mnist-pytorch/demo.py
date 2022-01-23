# Demo program using custom Pytorch wrapper code

import gpu_funcs as gf
import mlp
import plotting as pl
import train_eval as te

import torch

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms



if __name__ == '__main__':
    # Get dataset from directory
    dataset = MNIST(root='data', download=False, train=True, transform=transforms.ToTensor())
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

    pl.show_all_batch_imgs8x8(dl=train_loader)

    # Get MLP model
    mnist_model = mlp.MLP(n_features=28*28, n_outputs=10)

    # Switch to device, and move model and data to device
    device = gf.get_default_device()
    train_loader = gf.DeviceDataLoader(train_loader, device)
    test_loader = gf.DeviceDataLoader(test_loader, device)
    mnist_model = gf.move_to_device(mnist_model, device)

    # Train MLP model
    history = te.train_model(
        model=mnist_model,
        epochs=n_epochs,
        lr=0.01,
        train_loader=train_loader,
        val_loader=test_loader,
        opt_func=torch.optim.SGD
    )

    # Visualize loss and accuracy history
    pl.plot_accuracy(history)
    pl.plot_losses(history)

    # Evaluate final model
    scores = te.evaluate(model=mnist_model, val_loader=test_loader)
    print('Test scores: ', scores)

    # Predict on a few inputs
    test_dataset = MNIST(root='data/', train=False, transform=transforms.ToTensor())
    x, label = dataset[0]
    x = x.unsqueeze(0)
    pred = te.predict(x=x, model=mnist_model)
    print('True label: {}, Predicted: {}'.format(label, pred))

    x, label = dataset[111]
    x = x.unsqueeze(0)
    pred = te.predict(x=x, model=mnist_model)
    print('True label: {}, Predicted: {}'.format(label, pred))

    x, label = dataset[222]
    x = x.unsqueeze(0)
    pred = te.predict(x=x, model=mnist_model)
    print('True label: {}, Predicted: {}'.format(label, pred))

    x, label = dataset[333]
    x = x.unsqueeze(0)
    pred = te.predict(x=x, model=mnist_model)
    print('True label: {}, Predicted: {}'.format(label, pred))

    x, label = dataset[444]
    x = x.unsqueeze(0)
    pred = te.predict(x=x, model=mnist_model)
    print('True label: {}, Predicted: {}'.format(label, pred))

    x, label = dataset[555]
    x = x.unsqueeze(0)
    pred = te.predict(x=x, model=mnist_model)
    print('True label: {}, Predicted: {}'.format(label, pred))

    # Save model
    filename = 'mnist_model.pth'
    torch.save(mnist_model.state_dict(), filename)

    # Reload model
    mnist_model2 = mlp.MLP(n_features=28*28, n_outputs=10)
    mnist_model2.load_state_dict(torch.load(filename))
    scores = te.evaluate(model=mnist_model, val_loader=test_loader)
    print('Model 2 test scores: ', scores)

    print('Demo finished.')
