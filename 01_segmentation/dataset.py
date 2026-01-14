import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
import monai
import monai.transforms as mT

class CustomDataset(Dataset):
    def __init__(self,meta_df, resize=[256,128,128]):
        self.meta_df = meta_df
        self.anatomy = [0,1,2] # 0 = background, 1= canal, 2 = molar
        self.resize=resize #d, h,w
        self.ct_resizer = mT.Resize(self.resize)
        self.mask_resizer = mT.Resize(self.resize,mode='nearest')
        self.mode =mode
        self.cache={}
        
    def __len__(self):
        return len(self.meta_df)
    
    def pydiocm_dcmread(self,path):
        return pydicom.dcmread(path).pixel_array
    
    def min_max_norm_dcm(self, x):
        x[x>3071] = 3071
        return ((x - 0+1e-9)/(3071-0+1e-9)).astype(np.float32)
    
    def __getitem__(self,idx):
        sample_input = {}
        sample = self.meta_df.iloc[idx,:].to_dict()
        sample['cbct_path']= sorted(glob.glob(os.path.join(sample['cbct_dir'],'*')), key=lambda f: pydicom.dcmread(f).InstanceNumber)
        sample['mask_path']= sorted(glob.glob(os.path.join(sample['mask_dir'],'*')), key=lambda f: pydicom.dcmread(f).InstanceNumber)
        cbct = np.stack([self.pydiocm_dcmread(x) for x in sample['cbct_path']])
        mask = np.stack([self.pydiocm_dcmread(x) for x in sample['mask_path']])
        
        d,h,w = cbct.shape
        if sample['left_right'] =='left':
            cbct = cbct[:,:,:int(w/2)]
            mask = mask[:,:,:int(w/2)]
        elif sample['left_right'] =='right':
            cbct = cbct[:,:,int(w/2):]
            mask = mask[:,:,int(w/2):]
          
        input_ = np.expand_dims(self.min_max_norm_dcm(cbct),axis=0).astype(np.float32)
        target_ = np.stack([(mask==x).astype(np.uint8) for x in self.anatomy], axis=0)
        resize_cbct = self.ct_resizer(input_)
        resize_mask = self.mask_resizer(target_)
        
        sample_input['origin_size'] = input_.shape
        sample_input['input'] = resize_cbct
        sample_input['target'] = resize_mask

        return sample_input
