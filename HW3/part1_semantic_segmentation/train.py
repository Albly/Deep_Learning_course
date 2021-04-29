# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications Copyright Skoltech Deep Learning Course.

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .model import UNet, DeepLab
from .dataset import FloodNet
from . import loss


USE_WANDB = False 
#delete before submittion
#---------------------------------------------------------------
try:
    import wandb
    USE_WANDB = True
except ImportError:
    USE_WANDB = False
    print("Error with importing wandb. Using Tensorboard")
#---------------------------------------------------------------


class SegModel(pl.LightningModule):
    def __init__(
        self,
        model: str,
        backbone: str,
        aspp: bool,
        augment_data: bool,
        optimizer: str = 'default',
        scheduler: str = 'default',
        lr: float = None,
        batch_size: int = 16,
        data_path: str = 'datasets/tiny-floodnet-challenge',
        image_size: int = 256,
    ):
        super(SegModel, self).__init__()
        self.num_classes = 8

        if model == 'unet':
            self.net = UNet(self.num_classes)
        elif model == 'deeplab':
            self.net = DeepLab(backbone, aspp, self.num_classes)

        self.train_dataset = FloodNet(data_path, 'train', augment_data, image_size)
        self.test_dataset = FloodNet(data_path, 'test', augment_data, image_size)

        self.batch_size = batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.eps = 1e-7

        # Visualization
        self.color_map = torch.FloatTensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
             [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        pred = self.forward(img)

        train_loss = F.cross_entropy(pred, mask)

        #LOGGING 
        self.log('train_loss', train_loss, prog_bar=True)
#---------------------------------------------------------------
        if USE_WANDB:
            wandb.log({'Train_loss': train_loss})
#---------------------------------------------------------------

        return train_loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        pred = self.forward(img)

        #validation loss calcualation
        val_loss = F.cross_entropy(pred, mask)
        #LOGGING
        self.log("val_loss", val_loss, prog_bar = True)
#---------------------------------------------------------------
        if USE_WANDB:
            wandb.log({'val_loss': val_loss})
#---------------------------------------------------------------        

        intersection, union, target = loss.calc_val_data(pred, mask, self.num_classes)

        return {'intersection': intersection, 'union': union, 'target': target, 'img': img, 'pred': pred, 'mask': mask}

    def validation_epoch_end(self, outputs):
        intersection = torch.cat([x['intersection'] for x in outputs])
        union = torch.cat([x['union'] for x in outputs])
        target = torch.cat([x['target'] for x in outputs])

        mean_iou, mean_class_rec, mean_acc = loss.calc_val_loss(intersection, union, target, self.eps)

        #LOGGING
        log_dict = {'mean_iou': mean_iou, 'mean_class_rec': mean_class_rec, 'mean_acc': mean_acc}

        for k, v in log_dict.items():
            self.log(k, v, prog_bar=True)
#---------------------------------------------------------------   
        if USE_WANDB:
            wandb.log(log_dict)
#---------------------------------------------------------------   
        # Visualize results
        img = torch.cat([x['img'] for x in outputs]).cpu()
        pred = torch.cat([x['pred'] for x in outputs]).cpu()
        mask = torch.cat([x['mask'] for x in outputs]).cpu()

        pred_vis = self.visualize_mask(torch.argmax(pred, dim=1))
        mask_vis = self.visualize_mask(mask)

        results = torch.cat(torch.cat([img, pred_vis, mask_vis], dim=3).split(1, dim=0), dim=2)
        results_thumbnail = F.interpolate(results, scale_factor=0.25, mode='bilinear')[0]

        #LOGGING
        self.logger.experiment.add_image('results', results_thumbnail, self.current_epoch)
#--------------------------------------------------------------- 
        if USE_WANDB:
            wandb.log({"Images": [wandb.Image(results_thumbnail, caption='Epoch: '+str(self.current_epoch))]})
#--------------------------------------------------------------- 
    def visualize_mask(self, mask):
        b, h, w = mask.shape
        mask_ = mask.view(-1)

        if self.color_map.device != mask.device:
            self.color_map = self.color_map.to(mask.device)

        mask_vis = self.color_map[mask_].view(b, h, w, 3).permute(0, 3, 1, 2).clone()

        return mask_vis

    def configure_optimizers(self):
        # TODO: 2 points
        # Use self.optimizer and self.scheduler to call different optimizers
        if self.optimizer == 'default':
            opt = torch.optim.Adam(self.net.parameters(), lr = 1e-3)
        elif self.optimizer == 'Adam':
            opt = torch.optim.Adam(self.net.parameters(), lr = self.lr)
        elif self.optimizer == 'Adam_L2':
            opt = torch.optim.Adam(self.net.parameters(), lr = self.lr, weight_decay = 1e-4)
        else: 
            print('Wrong optimizer. Using default')
            opt = torch.optim.Adam(self.net.parameters(), lr = 1e-3)

        # TODO: init learning rate scheduler
        if self.scheduler == 'default':
            sch = torch.optim.lr_scheduler.MultiStepLR(opt, [20,30], gamma = 0.1) 
        elif self.scheduler == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
            sch = {'scheduler': lr_scheduler, 'interval': 'step', 'monitor': 'val_loss'}
        else:
            print('Wrong scheduler. Using default')
            sch = torch.optim.lr_scheduler.MultiStepLR(opt, [20,30], gamma = 0.1)

        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=8, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=8, batch_size=1, shuffle=False)