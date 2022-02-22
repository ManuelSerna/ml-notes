# Helper Functions for training
# for Pytorch projects
# ALIAS:

import time
import torch

import torch.nn as nn

import pytorch_wrapper.computations as comps
import pytorch_wrapper.gpu_funcs as gpu_funcs
import pytorch_wrapper.losses as losses


#===========================================
# Functions for simple Sequential models
#===========================================
@torch.no_grad()
def evaluate_sequential(model=None, val_loader=None):
    ''' Without modifying gradients, evaluate validation set of images to get average loss and accuracy
    (over all batches in DataLoader object)
    NOTE:

    :param model: torch.nn.Module model
    :param val_loader: validation data loader (type DataLoader)
    :return: averaged validation accuracies and averaged validation losses in a dictionary in the form
        {
            'val_acc': number
            'val_loss': number
        }
    '''
    model.eval() # set to evaluate mode, i.e., layers like batch norm and dropout will work
    val_losses = []
    val_accs = []
    val_scores = {}

    for batch in val_loader:
        val_loss = losses.compute_batch_ce_loss(model=model, batch=batch)
        inputs, labels = batch
        out = model(inputs)
        val_accuracy = comps.get_accuracy(out, labels)  # compute accuracy

        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

    val_scores['val_loss'] = comps.avg_list_tensors(val_losses)
    val_scores['val_acc'] = comps.avg_list_tensors(val_accs)

    return val_scores


def train_sequential_model(
        model=None,
        epochs=10,
        lr=0.01,
        lr_schedule=False,
        weight_decay=0,
        train_loader=None,
        val_loader=None,
        grad_clip=None,
        opt_func=torch.optim.SGD):
    ''' Train basic sequential model (e.g., MLP, CNN for MNIST, etc.)

    :param model: model object that inherits from torch.nn.Module
    :param epochs: (int) number of epochs to train for
    :param lr: max learning rate
    :param lr_schedule: (bool) enable learning rate scheduling (if True, parameter "lr" will be used as max lr)
    :param weight_decay: (float) value for weight decay (0 does nothing)
    :param train_loader: training DataLoader object
    :param val_loader: validation DataLoader object
    :param grad_clip: (number) limit how much gradients can change by this much (default = None)
    :param opt_func: optimization function (from torch.optim)
    :return: list of dictionaries, each element having the form for index/epoch i
        {
            'train_loss': number
            'val_acc': number
            'val_loss': number,
            'lrs': list
        }
    '''
    torch.cuda.empty_cache()

    history = []
    optimizer = opt_func(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    if lr_schedule:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs
        )

    start = time.time()

    for epoch in range(epochs):
        # Train
        model.train() # set to train mode
        train_losses = []
        lrs = []

        # Train on batches
        for batch in train_loader:
            loss = losses.compute_batch_ce_loss(model=model, batch=batch)
            train_losses.append(loss) # add loss of current batch
            loss.backward() # backpropagate on loss
            optimizer.step() # perform single optimization step
            optimizer.zero_grad() # reset gradients to zero for future backpropagation (useful for RNNs, but not here)

            # Clip gradients
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            # Update learning rate via scheduler
            if lr_schedule:
                lrs.append(sched.get_last_lr())
                sched.step()
            else:
                lrs.append(optimizer.param_groups[0]['lr']) # if not using scheduler, get lr from optimizer

        # Evaluate model after training for one additional epoch
        avg_train_loss = comps.avg_list_tensors(train_losses)
        val_scores = evaluate_sequential(model=model, val_loader=val_loader) # get validation accuracy and loss

        # Finally, add losses and validation accuracy of current epoch to history list
        epoch_scores = {
            'train_loss': avg_train_loss,
            'val_acc': val_scores['val_acc'],
            'val_loss': val_scores['val_loss'],
            'lrs': lrs
        }

        history.append(epoch_scores)
        print('Epoch {}/{}: Train loss={:.4f}, Val loss = {:.4f}, Val accuracy = {:.4f}'.format(
            epoch+1,
            epochs,
            avg_train_loss,
            val_scores['val_loss'],
            val_scores['val_acc']
        ))

    end = time.time() - start
    print('Total training time: {:.3f} min'.format(end/60))

    return history


def predict(x=None, model=None):
    ''' Use model to make prediction on input
    NOTE: prediction is highest value in prediction vector

    :param x: batch of images in tensor
    :param model: model object that inherits from torch.nn.Module
    :return: prediction label
    '''
    device = gpu_funcs.get_default_device()
    x = gpu_funcs.move_to_device(data=x, device=device)
    predict_vector = model(x)
    best_pred_values, best_pred_indices = torch.max(predict_vector, dim=1)
    return best_pred_indices[0].item()
