import torch
from torch import optim, cuda, nn
from torch.utils import data
from torch.distributions.normal import Normal
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torchinfo import summary
import torchio as tio
import functools
from sklearn.model_selection import KFold
from pathlib import Path

class training(model, model_dir, dataset, args):
  epoch_loss = []
  epoch_total_loss = []
  epoch_step_time = []
  device = 'cuda'
  torch.cuda.set_device(0)

  for fold, (train_indices,val_indices) in enumerate(splits.split(np.arange(len(dataset)))):
      if fold == 1:
          break
      print('Fold {}'.format(fold + 1))

      # Creating PT data samplers and loaders:
      train_sampler = SubsetRandomSampler(train_indices)
      valid_sampler = SubsetRandomSampler(val_indices)
      train_loader = data.DataLoader(dataset, batch_size=1, sampler=train_sampler)
      validation_loader = data.DataLoader(dataset, batch_size=1, sampler=valid_sampler)

      model.to(device)
      model.train()

      history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
      history_epoch_loss = []
      # training loops
      nb_epochs = args['nb_epochs']
      exa_ = 0
      for epoch in range(0, nb_epochs):
          print("Epoch: {}".format(epoch))
          # save model checkpoint
          if epoch % 20 == 0:
              torch.save(model.state_dict(),os.path.join(model_dir, '%04d.pt' % epoch))

          step_start_time = time.time()
          model.train()
          train_loss = [0.0, 0.0]

          y_pred_full = torch.zeros(inshape3D)
          # generate inputs (and true outputs) and convert them to tensors
          for y_src, target, y_true, name in train_loader:
              y_src = y_src.to(device).float()
              target = target.to(device).float()
              y_true = y_true.to(device).float()

              y_pred, y_bin_seg = model(y_src,target)
              target_mask = mask_AOI.tensor.to(device).float()

              # calculate total loss
              seg_loss = loss_seg(y_bin_seg,target_mask.unsqueeze(0))
              reg_loss = loss_reg(y_pred, target.unsqueeze(0))
              loss = [seg_loss, reg_loss]
              train_loss[0] += loss[0]
              train_loss[1] += loss[1]
              print(loss)
              epoch_loss.append(loss)

              # backpropagate and optimize
              optimizer.zero_grad()
              torch.autograd.backward(loss)
              optimizer.step()

          # get compute time
          history_epoch_loss.append(epoch_loss)
          train_loss[0] = train_loss[0] / len(train_loader)
          train_loss[1] = train_loss[1] / len(train_loader)
          epoch_step_time.append(time.time() - step_start_time)
          history['train_loss'].append(train_loss)

          ### VALIDATION
          print("---------------------------------------------START VALIDATION---------------------------------------------")
          valid_loss = [0.0, 0.0]
          model.eval()
          for y_src, target, y_true, name in validation_loader:
              with torch.no_grad():
                  y_src = y_src.cuda().float()
                  target = target.cuda().float()

                  y_pred, y_seg = model(y_src,target)
                  y_bin_seg = tio.ScalarImage(tensor=y_seg.cpu().squeeze(0))
                  y_bin_seg.plot()
                  target_mask = mask_AOI.tensor.to(device).float()

                  seg_loss = loss_seg(y_seg,target_mask.unsqueeze(0))
                  reg_loss = loss_reg(y_pred, target.unsqueeze(0))
                  loss = [seg_loss, reg_loss]
                  valid_loss[0] += loss[0]
                  valid_loss[1] += loss[1]

          valid_loss[0] = valid_loss[0] / len(train_loader)
          valid_loss[1] = valid_loss[1] / len(train_loader)
          history['test_loss'].append(valid_loss)
          # Leraning Rate Scheduler
          scheduler.step((valid_loss[0]+valid_loss[1])/2)
          
          # print epoch info
          print(f'Epoch {epoch} \t\t Training - MSELoss: {train_loss} \t\t Validation - MSELoss: {valid_loss}')
          epoch_info = 'Epoch %d/%d' % (epoch + 1, nb_epochs)
          time_info = '%.4f sec/step' % np.mean(epoch_step_time)
          print(' - '.join((epoch_info, time_info)), flush=True)

      # final model save
      torch.save(model.state_dict(),os.path.join(model_dir, 'final%04d.pt' % epoch))
      return model
