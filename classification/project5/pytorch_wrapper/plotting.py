# Plotting and other viewing-related helper functions
# for Pytorch projects

import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_tensor_image(img=None):
    ''' Use matplotlib to display image

    :param img: image as a tensor
    :return: NA
    '''
    plt.imshow(img.permute(1, 2, 0)) # switch axes to show img properly


def show_all_batch_imgs8x8(dl=None):
    ''' Show 8x8 grid of images from DataLoader object

    :param dl: DataLoader object
    :return: NA
    '''
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0).clamp(0, 1))
        break

    plt.show()


def plot_pred_img(img=None, ground_truth=None, predicted=None, savedir=''):
    ''' Plot data sample where corresponding output is an image

    :param img: (np.array) original image passed as input to model
    :param ground_truth: (np.array) ground-truth image (e.g., segmentation mask)
    :param predicted: (np.array) predicted image (e.g., segmentation mask)
    :param savedir: (string) directory to store figure in
    :return: NA
    '''
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    ax = axes.ravel()

    ax[0].imshow(img, vmin=0, vmax=1, interpolation='bicubic')
    ax[0].set_title('Original Image')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(ground_truth, vmin=0, vmax=1, interpolation='nearest')
    ax[1].set_title('Ground Truth')
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[2].imshow(predicted, vmin=0, vmax=1, interpolation='nearest')
    ax[2].set_title('Predicted')
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    plt.tight_layout()

    if savedir != '':
        fig.savefig(savedir, dpi=300)
    else:
        plt.show()

    plt.close()


def plot_accuracy(history=None):
    ''' Plot epochs vs accuracy

    :param history: list of dictionaries for losses and validation accuracy per epoch,
        each element having the form for index/epoch i
        {
            'train_loss': number
            'val_acc': number
            'val_loss': number
        }
    :return: NA
    '''
    val_accuracies = [x['val_acc'] for x in history]
    plt.plot(val_accuracies, '-rs', label='validation\naccuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [0.0, 1.0]')
    plt.ylim(0.000, 1.000)
    plt.legend(loc='best')
    plt.show()


def plot_losses(history=None):
    ''' Plot epochs vs training loss and validation loss

    :param history: list of dictionaries for losses and validation accuracy per epoch,
        each element having the form for index/epoch i
        {
            'train_loss': number
            'val_acc': number
            'val_loss': number
        }
    :return: NA
    '''
    train_losses  = [x['train_loss'] for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bs', label='training\nloss')
    plt.plot(val_losses, '-rs', label='validation\nloss')
    plt.title('Training and Validation Losses vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0.0)
    plt.legend(loc='best')
    plt.show()
