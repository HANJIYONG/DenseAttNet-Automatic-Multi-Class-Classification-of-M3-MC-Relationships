import os
import multiprocessing
import glob
from tqdm import tqdm
from natsort import natsorted
from easydict import EasyDict
import json
import numpy as np
import torch
import monai
import monai.transforms as mT
from skimage.measure import marching_cubes
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt


def pydiocm_dcmread(path):
    return pydicom.dcmread(path).pixel_array

def signed_distance_map(binary_image):
    # Calculate the Euclidean distance transform
    distance_map = distance_transform_edt(binary_image)

    # Invert the binary image
    inverted_image = np.logical_not(binary_image)

    # Calculate the Euclidean distance transform of the inverted image
    inverted_distance_map = distance_transform_edt(inverted_image)

    # Subtract the inverted distance map from the distance map
    signed_distance = distance_map - inverted_distance_map

    return signed_distance

def find_min_distance_points_kdtree(point_cloud1, point_cloud2):
    kdtree1 = cKDTree(point_cloud1)
    kdtree2 = cKDTree(point_cloud2)

    min_distance = float('inf')
    min_point_cloud1 = None
    min_point_cloud2 = None

    for point1 in point_cloud1:
        _, idx2 = kdtree2.query(point1)
        closest_point2 = point_cloud2[idx2]
        
        dist = np.linalg.norm(point1 - closest_point2)
        if dist < min_distance:
            min_distance = dist
            min_point_cloud1 = point1
            min_point_cloud2 = closest_point2

    return min_point_cloud1, min_point_cloud2, min_distance


device = torch.device('cuda')
model_segmentation = monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=1, out_channels=3, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
model_segmentation.load_state_dict(torch.load("01_segmentation/pretrained_wweight/model_statedict.pth",weights_only=False,map_location=torch.device('cpu')))
model_segmentation = model_segmentation.to(device)
model_segmentation.eval()

margin = 32
for i in tqdm(range(len(meta_df))):
  sample = meta_df.iloc[i].to_dict()
  sample['cbct_path']= sorted(glob.glob(os.path.join(sample['cbct_dir'],'*')), key=lambda f: pydicom.dcmread(f).InstanceNumber)
  
  with multiprocessing.Pool(processes=16) as pool:
      dicom = np.stack(pool.map(pydiocm_dcmread, sample['cbct_path']))    
  
  if sample['t_48']:
      molar_ = 48
      half_dcm = dicom[:,:,:mask.shape[-1]//2].copy()
      
      ct_resizer = mT.Resize([128, 256, 128])
      sample_cbct = half_dcm.copy().astype(np.float32)
      sample_cbct[sample_cbct>3071] = 3071
      sample_cbct /= 3071.
      sample_cbct = ct_resizer(sample_cbct[None])
      
      with torch.no_grad():
          pred_mask = model_segmentation(sample_cbct[None].to(device)).squeeze().argmax(0).detach().cpu()
      pred_mask = mask_resizer(pred_mask[None]).squeeze().numpy()
  
      set1 = torch.tensor(np.stack(np.where(pred_mask==1),axis=0).T).float() # canal
      set2 = torch.tensor(np.stack(np.where(pred_mask==2),axis=0).T).float() # molar
  
      min_point1, min_point2, min_dist = find_min_distance_points_kdtree(set1, set2)
  
      center_point = ((min_point1+min_point2)/2).int()
      c_z,c_y,c_x = center_point
  
      roi_dcm = half_dcm[c_z-margin:c_z+margin, c_y-margin:c_y+margin, c_x-margin:c_x+margin]
      roi_pred = pred_mask[c_z-margin:c_z+margin,c_y-margin:c_y+margin, c_x-margin:c_x+margin]
      roi_sdm = signed_distance_map(roi_mask)
      
      temp_dict4save = {}
      temp_dict4save['roi_dcm'] = roi_dcm
      temp_dict4save['roi_mask_pred'] = roi_pred
      temp_dict4save['roi_sdm'] = roi_sdm
      temp_dict4save['margin'] = margin
      temp_dict4save['half_dcm_shape'] = half_dcm.shape
      temp_dict4save['center_zyx'] = c_z.item(),c_y.item(),c_x.item()
      dst = f"{sample['patient'].strip()}_{molar_}_{sample['class'].strip()}.npy"
      
      np.save(os.path.join(roi_direc,dst),temp_dict4save)
      
  if sample['t_38']:
      molar_ = 38
      half_dcm = dicom[:,:,mask.shape[-1]//2:].copy()
      ct_resizer = mT.Resize([128, 256, 128])
      sample_cbct = half_dcm.copy().astype(np.float32)
      sample_cbct[sample_cbct>3071] = 3071
      sample_cbct /= 3071.
      sample_cbct = ct_resizer(sample_cbct[None])
      
      with torch.no_grad():
          pred_mask = model_segmentation(sample_cbct[None].to(device)).squeeze().argmax(0).detach().cpu()
      pred_mask = mask_resizer(pred_mask[None]).squeeze().numpy()
  
      set1 = torch.tensor(np.stack(np.where(pred_mask==1),axis=0).T).float() # canal
      set2 = torch.tensor(np.stack(np.where(pred_mask==2),axis=0).T).float() # molar
  
      min_point1, min_point2, min_dist = find_min_distance_points_kdtree(set1, set2)
  
      center_point = ((min_point1+min_point2)/2).int()
      c_z,c_y,c_x = center_point
  
      roi_dcm = half_dcm[c_z-margin:c_z+margin, c_y-margin:c_y+margin, c_x-margin:c_x+margin]
      roi_pred = pred_mask[c_z-margin:c_z+margin,c_y-margin:c_y+margin, c_x-margin:c_x+margin]
      roi_sdm = signed_distance_map(roi_mask)
      
      temp_dict4save = {}
      temp_dict4save['roi_dcm'] = roi_dcm
      temp_dict4save['roi_mask_pred'] = roi_pred
      temp_dict4save['roi_sdm'] = roi_sdm
      temp_dict4save['margin'] = margin
      temp_dict4save['half_dcm_shape'] = half_dcm.shape
      temp_dict4save['center_zyx'] = c_z.item(),c_y.item(),c_x.item()
      dst = f"{sample['patient'].strip()}_{molar_}_{sample['class'].strip()}.npy"
      
      np.save(os.path.join(roi_direc,dst),temp_dict4save)
      
