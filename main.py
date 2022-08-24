#!/usr/bin/env python

import os
import random
import argparse
import time
import numpy as np
import torch
from torchinfo import summary

import file_reader
import msot_dataloader
import MSOT_3DReg
import losses
import trainer
import tester

paths = {
    'mri': 'path/to/file',
    'mask': 'path/to/file',
    'label': 'path/to/file',
    'input': 'path/to/file',
    'model': 'path/to/file'
}
args = {
    'inshape3D': (96,48,112),
    'nb_epochs': 10,
}
Paras = {
    'patch_dims': [64,64,64],
    'patch_label_dims': [64, 64, 64],
    'patch_strides': [16, 16, 16],
    'n_class': 2,
    'batch_size': 1,
    'shuffle': False,
    'num_workers': 1,
}
np.random.seed(42)
val_per = 0.2
k = 10
torch.manual_seed(42)
loss_seg = losses.RMSE()
loss_reg = losses.dice() #Is the usual loss metric, see 2D implementation

msot_data, msot_label, mri_data, file_list = file_reader.reader(args['inshape3D'], paths['mri'], paths['mask'], paths['label'], paths['input'])
dataset = msot_dataloader.Dataset(filenumber, mri_data, msot_data, msot_label, file_list, inshape)

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
print(dataset_size)
indices = list(range(dataset_size))
split = int(np.floor(val_per * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Create model
model = MSOT_3DReg.MSOT_3DReg(
        inshape=args['inshape3D']
    )
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
# print overview
print(summary(model, input_size=((1,1,96,48,112),(1,1,96,48,112))))

model = trainer.training(model, paths['model'], dataset, args)
