import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    def __init__(self,meta_df):
        self.meta_df = meta_df
        
    def __len__(self):
        return len(self.meta_df)
      
    def normalize_dcm(self,x):
        arr = np.clip(x, -1000, 3071)
        arr = (arr-(-1000)) / (3071-(-1000))
        return arr
        
    def __getitem__(self,idx):
        sample = self.meta_df.iloc[idx,:].to_dict()
        roi = np.load(sample['path'],allow_pickle=True).item()
        roi_volume = roi['roi_dcm']
        roi_sdm = roi['roi_sdm']
        
        roi_volume = self.normalize_dcm(roi_volume)
        roi_sdm = (roi_sdm-roi_sdm.min())/(roi_sdm.max()-roi_sdm.min())
        
        roi_volume = np.expand_dims(roi_volume,axis=0)
        roi_sdm = np.expand_dims(roi_sdm,axis=0)
        roi_input = np.concatenate([roi_volume,roi_sdm],axis=0)
      
        label = torch.tensor(int(sample['class']))
        
        sample['input'] = roi_input
        sample['label'] = label
            
        return sample
        
