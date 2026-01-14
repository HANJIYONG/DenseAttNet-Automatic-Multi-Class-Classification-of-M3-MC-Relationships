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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ClassificationDataset

from model import DenseAttnNet

import monai
from modules.earlystop import EarlyStopping


def train(h_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = pd.read_csv('metadata_train.csv')
    val_df = pd.read_csv('metadata_val.csv')
    
    train_dataset = ClassificationDataset(train_df)
    val_dataset = ClassificationDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=h_params.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=h_params.batch_size, shuffle=False)

    model = DenseAttnNet(spatial_dims= 3, in_channels= 2, out_channels=5, block_config= (6, 12, 24, 16))
    model = model.to(device)
    
    optim = torch.optim.AdamW(params=model.parameters(),lr = h_params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,patience=h_params.lr_schedule_patience)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
      
    monitor = EarlyStopping(patience=h_params.earlystop_patience, verbose=True, path=h_params.model_save_path)
    
    metric_logger = {k:[] for k in ['train_loss','val_loss',
                                'train_acc','val_acc',
                                'lr']}

    total_train_num = len(train_loader.sampler)
    total_val_num = len(val_loader.sampler)
    
    for epoch in range(h_params.total_epoch):
        
        for param in optim.param_groups:
            lr_stauts = param['lr']
        metric_logger['lr'].append(lr_stauts)
    
        epoch_loss = {k:0 for k in metric_logger if k not in ['lr']}
        train_gt, train_pred =[],[]
        val_gt, val_pred =[],[]
        
        print(f"Epoch {epoch+1:03d}/{h_params.total_epoch:03d}\tLR: {lr_stauts:.0e}")
        
        model.train()
        for data in tqdm(train_loader,total=len(train_loader),position=0,desc='Train',colour='blue'):
            batch_num = len(data['input'])
            
            image = data['input'].to(device)
            target = data['label'].to(device)
            
            pred = model(image.float())
            loss = criterion(pred,target.flatten())
            
            train_gt.append(target.cpu())
            train_pred.append(pred.cpu())
          
            optim.zero_grad()
            loss.backward()
            optim.step()
                  
            epoch_loss['train_loss'] += loss.item()*batch_num
            
        model.eval()
        with torch.no_grad():
            for data in tqdm(val_loader,total=len(val_loader),position=0,desc='Val',colour='green'):
                batch_num = len(data['input'])
            
                image = data['input'].to(device)
                target = data['label'].to(device)
    
                pred = model(image.float())
                loss = criterion(pred,target.flatten())
                
                val_gt.append(target.cpu())
                val_pred.append(pred.cpu())
    
                epoch_loss['val_loss'] += loss.item()*batch_num
              
        epoch_loss = {k:(v/total_train_num if 'train' in k else v/total_val_num) for k,v in epoch_loss.items()}
        
        for k,v in epoch_loss.items():
            if 'acc' in k:
                continue
        
            metric_logger[k].append(v)
            
        # multi class version
        metric_logger['train_acc'].append(((torch.concatenate(train_gt) == torch.concatenate(train_pred).argmax(dim=1)).float().mean()).item())
        metric_logger['val_acc'].append(((torch.concatenate(val_gt) == torch.concatenate(val_pred).argmax(dim=1)).float().mean()).item())
    
        monitor(epoch_loss['val_loss'],model)
        if monitor.early_stop:
            print(f"Train early stopped, Minimum validation loss: {monitor.val_loss_min}")
            break
        
        scheduler.step(epoch_loss['val_loss'])
      
        print(f"Train loss: {epoch_loss['train_loss']:.7f}\tTrain ACC: {metric_logger['train_acc'][-1]:.4f}\n\ 
        Val loss: {epoch_loss['val_loss']:.7f}\tVal ACC: {metric_logger['val_acc'][-1]:.4f}")
    
        with open(os.path.join(h_params.trial_path,'metric_logger.json'),'w') as f:
            json.dump(metric_logger, f)
    


if __name__ == "__main__":
  h_params=EasyDict()
  
  h_params.total_epoch=500 #training epochs
  h_params.batch_size=4 #batch size
  h_params.lr=1e-4 #batch size
  h_params.lr_schedule_patience=20 #ReduceOnPlateau
  h_params.earlystop_patience=40 #early stop patience
  
  h_params.size=[64,64,64]
  
  h_params.model_name = "model.pth"
  h_params.model_save_base = os.path.join(os.getcwd(),"train_log")
  now = time.localtime(time.time())
  h_params.trial_ = f"{now.tm_year}{now.tm_mon:02d}{now.tm_mday:02d}_{now.tm_hour:02d}{now.tm_min:02d}{now.tm_sec:02d}"
  h_params.trial_path =os.path.join(h_params.model_save_base,h_params.trial_)
  h_params.model_save_path = os.path.join(h_params.trial_path,h_params.model_name)
  
  train(h_params)
