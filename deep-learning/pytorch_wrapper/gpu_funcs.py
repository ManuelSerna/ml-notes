# Helper functions for GPU-related things
# for Pytorch projects
# ALIAS: gpu_funcs

import torch


def get_default_device():
    ''' Check if GPU is available

    :return: NA
    '''
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def move_to_device(data=None, device=None):
    ''' Move individual tensor to chosen device
    (since we cannot do this for the DataLoader object itself,
    we must move the input data itself to the device)

    :param data: dataset as Tensor
    :param device: device (i.e., whether CUDA is available)
    :return: Data migrated to device
    '''
    if isinstance(data, (list, tuple)):
        return [move_to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    ''' Wrap DataLoader object to move data to designated device'''

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        ''' Yield batch of data after moving it to device'''
        for b in self.dl:  # iterate over samples in batch
            yield move_to_device(b, self.device)  # return next item, which is in appropriate device

    def __len__(self):
        ''' Number of batches'''
        return len(self.dl)
