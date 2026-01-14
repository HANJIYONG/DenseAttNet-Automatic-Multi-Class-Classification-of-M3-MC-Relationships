import time
import os
import multiprocessing
import glob
import json
from easydict import EasyDict
from natsort import natsorted
from tqdm import tqdm

import numpy as np
import pandas as pd
import pydicom

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import SegmentationDataset
from torch.utils.data import DataLoader
import monai

from modules.loss import FocalLoss,dice_loss,DiceChannelLoss
from modules.earlystop import EarlyStopping


def train(h_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = pd.read_csv('metadata_train.csv')
    val_df = pd.read_csv('metadata_val.csv')
    
    train_dataset = ClassificationDataset(train_df)
    val_dataset = ClassificationDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=h_params.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=h_params.batch_size, shuffle=False)

    model = monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=1, out_channels=3, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
    model = model.to(device)
    
    optim = torch.optim.AdamW(params=model.parameters(),lr = h_params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,patience=h_params.lr_schedule_patience)
  
    criterion_focal = FocalLoss(reduction='mean')
    def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
      return torch.softmax(x, 1)
    criterion_SoftSkeletonRecal = SoftSkeletonRecallLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
  
    monitor = EarlyStopping(patience=h_params.earlystop_patience, verbose=True, path=h_params.model_save_path)

    
    metric_logger = {k:[] for k in ['train_focal','val_focal', 'train_dice','val_dice', 'train_skeleton','val_skeleton', 'train_loss','val_loss', 'lr']}

    total_train_num = len(train_loader.sampler)
    total_val_num = len(val_loader.sampler)

    for epoch in range(h_params.total_epoch):
        
        for param in optim.param_groups:
            lr_stauts = param['lr']
        metric_logger['lr'].append(lr_stauts)
    
        epoch_loss = {k:0 for k in metric_logger if k not in ['lr']}
        
        print(f"Epoch {epoch+1:03d}/{h_params.total_epoch:03d}\tLR: {lr_stauts:.0e}")
        model.train()
        for data in tqdm(train_loader,total=len(train_loader),position=0,desc='Train',colour='blue'):
            optim.zero_grad()
            
            batch_num = len(data['input'])
            
            image = data['input'].to(device)
            target = data['target'].to(device)
            
            pred = model(image)
            
            loss_dice = dice_loss(pred,target.float())
            loss_focal = criterion_focal(pred,target.float())
            loss_skeleton = criterion_SoftSkeletonRecal(pred,target.float())
            
            loss = loss_dice +loss_focal + loss_skeleton
            
            loss.backward()
            optim.step()
          
            epoch_loss['train_focal'] += loss_focal.item()*batch_num
            epoch_loss['train_dice'] += loss_dice.item()*batch_num
            epoch_loss['train_skeleton'] += loss_skeleton.item()*batch_num
            
            epoch_loss['train_loss'] += loss.item()*batch_num
            
        model.eval()
        with torch.no_grad():
            for data in tqdm(val_loader,total=len(val_loader),position=0,desc='Val',colour='green'):
                batch_num = len(data['input'])
            
                image = data['input'].to(device)
                target = data['target'].to(device)
    
                pred = model(image)
    
                loss_dice = dice_loss(pred,target.float())
                loss_focal = criterion_focal(pred,target.float())
                loss_skeleton = criterion_SoftSkeletonRecal(pred,target.float())
    
                loss = loss_dice +loss_focal + loss_skeleton
    
                epoch_loss['val_focal'] += loss_focal.item()*batch_num
                epoch_loss['val_dice'] += loss_dice.item()*batch_num
                epoch_loss['val_skeleton'] += loss_skeleton.item()*batch_num
                epoch_loss['val_loss'] += loss.item()*batch_num
        
        epoch_loss = {k:(v/total_train_num if 'train' in k else v/total_val_num) for k,v in epoch_loss.items()}
        
        for k,v in epoch_loss.items():
            metric_logger[k].append(v)
        
        monitor(epoch_loss['val_loss'],model)
        
        if monitor.early_stop:
            print(f"Train early stopped, Minimum validation loss: {monitor.val_loss_min}")
            break
        
        scheduler.step(epoch_loss['val_loss'])
        
        print(f"Epoch {epoch+1:03d}/{h_params.total_epoch:03d}\tLR: {lr_stauts:.0e}\n\
        Train loss: {epoch_loss['train_loss']:.7f}\tTrain focal: {epoch_loss['train_focal']:.7f}\tTrain dice: {epoch_loss['train_dice']:.7f}\tTrain skeleton: {epoch_loss['train_skeleton']:.7f}\n\
        Val loss: {epoch_loss['val_loss']:.7f}\tVal focal: {epoch_loss['val_focal']:.7f}\tVal dice: {epoch_loss['val_dice']:.7f}\tVal skeleton: {epoch_loss['val_skeleton']:.7f}")
        
        with open(os.path.join(h_params.trial_path,'metric_logger.json'),'w') as f:
            json.dump(metric_logger, f)

if __name__ == "__main__":
  h_params=EasyDict()
  h_params.total_epoch=500 #training epochs
  h_params.batch_size=2 #batch size
  h_params.lr=1e-4 #batch size
  h_params.lr_schedule_patience=10 #ReduceOnPlateau
  h_params.earlystop_patience=20 #early stop patience
  
  h_params.size=[128,256,128] #resize input data size(d,h,w)
  
  h_params.norm_value=[0.5,0.5] #normalize value(mean,std)
  
  h_params.model_name = "model.pth"
  h_params.model_save_base = os.path.join(os.getcwd(),"train_log")
  now = time.localtime(time.time())
  h_params.trial_ = f"{now.tm_year}{now.tm_mon:02d}{now.tm_mday:02d}_{now.tm_hour:02d}{now.tm_min:02d}{now.tm_sec:02d}"
  h_params.trial_path =os.path.join(h_params.model_save_base,h_params.trial_)
  h_params.model_save_path = os.path.join(h_params.trial_path,h_params.model_name)
  
  train(h_params)
