# Demo program using custom Pytorch wrapper code

import cnn
import pytorch_wrapper.display as disp
import pytorch_wrapper.gpu_funcs as gpu_funcs
import pytorch_wrapper.train_eval as te

import tarfile
import torch

from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

if __name__ == '__main__':
    # Get dataset
    ''' The CIFAR10 dataset has to be downloaded, AWS contains the .tar file,
    which of course has to be extracted.
    '''
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    local_data_dir = './data/cifar10'
    download_url(url=dataset_url, root='.')

    with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
        tar.extractall(path='./data')

    # Basic transform (since we should be working with tensors and not something like a PIL object)
    #transforms = transforms.Compose([transforms.ToTensor()])
    mean = [0.485, 0.456, 0.406] # statistics supposedly computed from ImageNet
    stdev = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdev, inplace=True)
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdev, inplace=True)
    ])

    # Load dataset objects
    train_ds = ImageFolder(root=local_data_dir+'/train', transform=train_transforms)
    test_ds = ImageFolder(root=local_data_dir+'/test', transform=val_transforms)

    # Get DataLoader objects (without worrying about device yet)
    batch_size = 256
    channels = 3

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0, # it seems on my machine, setting this >0 causes deadlock problems (I think...)
        pin_memory=True
    )

    #disp.show_all_batch_imgs8x8(dl=train_loader)

    # Get MLP model
    model = cnn.CIFAR10CNN(in_channels=channels, n_outputs=10)

    # Switch to device, and move model and data to device
    device = gpu_funcs.get_default_device()
    train_loader = gpu_funcs.DeviceDataLoader(train_loader, device)
    test_loader = gpu_funcs.DeviceDataLoader(test_loader, device)
    model = gpu_funcs.move_to_device(model, device)

    # Train MLP model
    lr = 0.01
    weight_decay = 0.0001
    n_epochs = 10
    grad_clip = 0.0001
    opt_func = torch.optim.Adam

    history = te.train_sequential_model(
        model=model,
        epochs=n_epochs,
        lr=lr,
        lr_schedule=True,
        weight_decay=weight_decay,
        train_loader=train_loader,
        val_loader=test_loader,
        grad_clip=grad_clip,
        opt_func=opt_func
    )

    # Visualize loss and accuracy history
    disp.plot_accuracy(history)
    disp.plot_losses(history)
    disp.plot_lrs(history)

    # Evaluate final model
    # TODO: find out why data loader takes absurdly long when called here :(
    scores = te.evaluate_sequential(model=model, val_loader=test_loader)
    print('Test scores: ', scores)

    # Predict on a few inputs
    x, label = test_ds[0]
    x = x.unsqueeze(0)
    pred = te.predict(x=x, model=model)
    print('True label: {}, Predicted: {}'.format(label, pred))

    x, label = test_ds[1111]
    x = x.unsqueeze(0)
    pred = te.predict(x=x, model=model)
    print('True label: {}, Predicted: {}'.format(label, pred))

    x, label = test_ds[2222]
    x = x.unsqueeze(0)
    pred = te.predict(x=x, model=model)
    print('True label: {}, Predicted: {}'.format(label, pred))

    x, label = test_ds[3333]
    x = x.unsqueeze(0)
    pred = te.predict(x=x, model=model)
    print('True label: {}, Predicted: {}'.format(label, pred))

    x, label = test_ds[4444]
    x = x.unsqueeze(0)
    pred = te.predict(x=x, model=model)
    print('True label: {}, Predicted: {}'.format(label, pred))

    x, label = test_ds[5555]
    x = x.unsqueeze(0)
    pred = te.predict(x=x, model=model)
    print('True label: {}, Predicted: {}'.format(label, pred))

    # Save model
    filename = 'cnn_model.pth'
    torch.save(model.state_dict(), filename)

    print('Demo finished.')
