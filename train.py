import numpy as np
from torch import optim, cuda, nn
from time import time
from copy import deepcopy
import gc
import cv2
import segmentation_models_pytorch as smp
from functools import partial

# import created reg_net from separate file
import reg_net
import preproc

# parse arguments
def parse_arg():
    parser = argparse.ArgumentParser()
    batch_size = 128
    nepochs = 300
    nworkers = 16
    seed = 1


    return batch_size, nepochs, nworkers, seed

# one step of training
def train(net, loader, criterion, optimizer, cuda):
    net.train()
    running_loss = 0
    running_accuracy = 0
    print('total batch:', len(loader))
    for i, (X, y) in enumerate(loader):
        if cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X), Variable(y)

        output = net(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        # get the index of the max log-probability
        _, pred = torch.max(output.data, 1)
        #pred = output.data.max(1, keepdim=True)[1]
        acc = pred.eq(y.data.view_as(pred)).cpu().sum()
        running_accuracy += acc
        print('batch/', i, '|loss/', loss.data[0], '|acc/', acc)
    return running_loss / len(loader), running_accuracy / len(loader.dataset)

def validate(net, loader, criterion, cuda):
    net.eval()
    running_loss = 0
    running_accuracy = 0

    for i, (X, y) in enumerate(loader):
        if cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X, volatile=True), Variable(y)
        output = net(X)
        loss = criterion(output, y)
        running_loss += loss.data[0]
        # get the index of the max log-probability
        _, pred = torch.max(output.data, 1)
        #pred = output.data.max(1, keepdim=True)[1]
        running_accuracy += pred.eq(y.data.view_as(pred)).cpu().sum()
    return running_loss / len(loader), running_accuracy / len(loader.dataset)

#full pipeline
def main():
    # INPUTS
    batch_size, nepochs, nworkers, seed  = parse_arg()
    cuda = torch.cuda.is_available()  # use cuda

    # BEGIN REGISTRATION
    if preprocess == True:
        preproc(obj)

    net = reg_net.Reg_Net.__init__
    criterion = torch.nn.BCELoss(reduction='sum') / batch_size #according to Hu et al. 2022

    optimizer = optim.Adam(net.parameters())

    for e in range(nepochs):
        start = time.time()
        train_loss, train_acc = train(net, train_loader,
            criterion, optimizer, cuda)
        val_loss, val_acc = validate(net, val_loader, criterion, cuda)
        end = time.time()

        # print stats
        stats ="""Epoch: {}\t train loss: {:.3f}, train acc: {:.3f}\t
                val loss: {:.3f}, val acc: {:.3f}\t
                time: {:.1f}s""".format( e, train_loss, train_acc, val_loss,
                val_acc, end-start)
        print(stats)
        print(stats, file=logfile)

        #early stopping and save best model
        if val_loss < best_loss:
            best_loss = val_loss
            utils.save_model({
                'arch': model,
                'state_dict': net.state_dict()
            }, 'saved-models/{}-run-{}.pth.tar'.format(model, run))

if __name__ == "__main__":

    print("start")
    main()
    print("Done!")
