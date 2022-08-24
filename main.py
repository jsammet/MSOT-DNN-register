#!/usr/bin/env python

import os
import random
import argparse
import time
import numpy as np
import torch

import file_reader
import msot_dataloader
import MSOT_3DReg
import losses

args = {
    'inshape': (112,128),
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
loss_seg = losses.RMSE() #nn.BCELoss() # diceloss() #nn.MSELoss() #
loss_reg = losses.dice()

msot_data, msot_label, mri_data, file_list = file_reader()

splits=KFold(n_splits=k,shuffle=True,random_state=42)
model = MSOT_3DReg(
        inshape=inshape3D
    )

