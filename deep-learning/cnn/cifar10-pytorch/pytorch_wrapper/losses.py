# Helper functions for computing losses
# for Pytorch projects
# ALIAS: losses

import torch.nn.functional as F


def compute_batch_ce_loss(model=None, batch=None):
    ''' Compute cross-entropy loss on batch of data
    NOTE: uses cross entropy loss with no modifications to parameters

    :param model: torch.nn.Module model
    :param batch: batch of tensors
    :return: loss object
    '''
    inputs, labels = batch
    out = model(inputs)  # get predictions
    loss = F.cross_entropy(input=out, target=labels)  # compute loss
    return loss
