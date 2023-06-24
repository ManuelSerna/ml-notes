# Helper Functions for training a generic segmentation model
# for Pytorch projects

import pytorch_wrapper.computations as comps
import pytorch_wrapper.gpu_funcs as gpu_funcs
import pytorch_wrapper.plotting as pl

import os.path
import time
import torch
import torch.nn as nn


def generic_segment_predict(dl_loader=None, model=None, epoch=None):
    ''' Generic method for having model predict batch of images

    :param dl_loader: DataLoader wrapper object
    :param model: model object that inherits from torch.nn.Module
    :param epoch: current epoch--used to label
    :return: prediction masks
    '''
    savedir = 'prediction_figs'
    samples_per_epoch = 6
    count = 0

    if not os.path.isdir(savedir):
        print('[INFO] Creating model outputs directory "{}"'.format(savedir))
        os.mkdir(savedir)

    for batch in dl_loader:
        batch_size = len(batch)
        imgs, masks = batch

        device = gpu_funcs.get_default_device()
        imgs = gpu_funcs.move_to_device(data=imgs, device=device)

        pred_masks = model(imgs)

        for i in range(batch_size):
            img = imgs[i].cpu().permute(1, 2, 0).detach().numpy() # original image
            mask = masks[i][0].cpu().detach().numpy() # original binary mask
            pred = pred_masks[i][0].cpu().detach().numpy() # predicted binary mask

            label = 'epoch_{}_img_{}'.format(epoch+1, i)
            pl.plot_pred_img(img, mask, pred, os.path.join(savedir, label))

        count += batch_size

        if count >= samples_per_epoch:
            break


@torch.no_grad()
def generic_evaluate_segmentation(model=None, val_loader=None, loss_func=None):
    ''' Without modifying gradients, evaluate validation set of images to get average loss and accuracy
    (over all batches in DataLoader object)
    NOTE: Uses cross-entropy loss

    :param model: torch.nn.Module model
    :param val_loader: validation data loader (type DataLoader)
    :param loss_func: Pytorch loss function
    :return: averaged validation accuracies and averaged validation losses in a dictionary in the form
        {
            'val_acc': number
            'val_loss': number
        }
    '''
    model.eval() # set to evaluate mode, i.e., layers like batch norm and dropout will work
    val_losses = []
    val_scores = {}

    for batch in val_loader:
        inputs, labels = batch
        out = model(inputs)
        val_loss = loss_func(input=out, target=labels)

        val_losses.append(val_loss)

    val_scores['val_loss'] = comps.avg_list_tensors(val_losses)

    return val_scores


def generic_train_segmentation_model(
    model=None,
    epochs=10,
    lr=1e-8,
    lr_schedule=False,
    weight_decay=0,
    train_loader=None,
    val_loader=None,
    grad_clip=None,
    loss_func=None,
    opt_func=torch.optim.RMSprop,
    weights_dir='weights_files'):
    ''' Train basic segmentation model (e.g., U-Net)
        NOTE: uses binary cross-entropy loss

        :param model: model object that inherits from torch.nn.Module
        :param epochs: (int) number of epochs to train for
        :param lr: max learning rate
        :param lr_schedule: (bool) enable learning rate scheduling (if True, parameter "lr" will be used as max lr)
        :param weight_decay: (float) value for weight decay (0 does nothing)
        :param train_loader: training DataLoader object
        :param val_loader: validation DataLoader object
        :param grad_clip: (number) limit how much gradients can change by this much (default = None)
        :param loss_func: Pytorch loss function
        :param opt_func: optimization function (from torch.optim)
        :param weights_dir: name of directory for weights files
        :return: list of dictionaries, each element having the form for index/epoch i
            {
                'train_loss': number
                ###'val_acc': number
                'val_loss': number,
                'lrs': list
            }
    '''
    torch.cuda.empty_cache()

    history = []
    optimizer = opt_func(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9
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
        model.train() # set to train mode
        train_losses = []
        lrs = []

        # Train on batches
        for batch in train_loader:
            inputs, labels = batch
            out = model(inputs)
            loss = loss_func(input=out, target=labels)

            train_losses.append(loss) # add loss of current batch

            optimizer.zero_grad() # reset gradients to zero for future backpropagation (useful for RNNs, but not here)
            loss.backward() # backpropagate on loss
            optimizer.step() # perform single optimization step

            # Clip gradients
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            # Update learning rate via scheduler
            if lr_schedule:
                lrs.append(sched.get_last_lr())
                sched.step()
            else:
                lrs.append(optimizer.param_groups[0]['lr'])  # if not using scheduler, get lr from optimizer

        # Evaluate model after training for one epoch
        avg_train_loss = comps.avg_list_tensors(train_losses)
        val_scores = generic_evaluate_segmentation(model=model, val_loader=val_loader, loss_func=loss_func) # get validation loss

        # Finally, add losses and validation accuracy of current epoch to history list
        epoch_scores = {
            'train_loss': avg_train_loss,
            'val_loss': val_scores['val_loss'],
            'lrs': lrs
        }

        # Save current model every so epochs and showcase output
        #if epoch % 5 == 0:
        if True:
            generic_segment_predict(val_loader, model, epoch)

            if not os.path.isdir(weights_dir):
                print('[INFO] Creating weights directory "{}"'.format(weights_dir))
                os.mkdir(weights_dir)

            weights_label = 'weights_epoch_{}.pth'
            torch.save(model.state_dict(), os.path.join(weights_dir, weights_label.format(epoch+1)))

        # Summarize epoch
        history.append(epoch_scores)
        print('Epoch {}/{}: Train loss={:.4f}, Val loss = {:.4f}'.format(
            epoch + 1,
            epochs,
            avg_train_loss,
            val_scores['val_loss']
        ))

    end = time.time() - start
    print('Total training time: {:.3f} min'.format(end / 60))

    return history
