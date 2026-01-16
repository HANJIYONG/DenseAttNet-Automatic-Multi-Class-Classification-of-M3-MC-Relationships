import os
import glob
import argparse
import torch
import numpy as np
import pydicom
import monai
import monai.transforms as mT
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

from Classification.model import DenseAttnNet
from utils.roi_crop import signed_distance_map, find_min_distance_points_kdtree


def preprocess_for_seg(image):
    img = image.copy()
    img[img > 3071] = 3071
    img /= 3071.
    resizer = mT.Resize([128, 256, 128]) 
    return resizer(img[None])

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Processing: {args.input_path}")
    print(f"[*] Using device: {device}")
    
    seg_model = monai.networks.nets.AttentionUnet(
        spatial_dims=3, in_channels=1, out_channels=3, 
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)
    ).to(device)

    seg_model_path = "pretrained_weights/mAttUNet_statedict.pth"
    seg_model.load_state_dict(torch.load(seg_model_path, map_location=device, weights_only=True))
    seg_model.eval()

    class_model = DenseAttnNet(
        spatial_dims=3, in_channels=2, out_channels=5, block_config=(6, 12, 24, 16)
    ).to(device)
    class_model_path = "pretrained_weights/DenseAttNet_statedict.pth"
    class_model.load_state_dict(torch.load(class_model_path, map_location=device, weights_only=True))
    class_model.eval()

    segmentation_transforms = mT.Compose([mT.Resize(spatial_size=[128,256,128]), mT.ToTensor()])
    classification_transforms = mT.Compose([mT.Resize(spatial_size=[64,64,64]), mT.ToTensor()])
    
    sample_input = {}
    full_dicom = nib.load(args.input_path).get_fdata().transpose(2,1,0)[::-1]
    d,h,w = full_dicom.shape
    
    # NOTE: The script attempts to parse the tooth number (#38 or #48) from the directory name.
    # In clinical scenarios where the target tooth is not pre-specified in the metadata, 
    # the full CBCT can be split into left and right hemispheres, and the inference 
    # process should be executed twice (once for each side) to evaluate both M3-MC regions.
    sample_tnum = args.input_path.split("_")[-1].split(".")[0]
    if sample_tnum =='48':
        half_dcm = full_dicom[:,:,:int(w/2)]
    elif sample_tnum =='38':
        half_dcm = full_dicom[:,:,int(w/2):]
    half_dcm+=1000

    input_seg = preprocess_for_seg(half_dcm).to(device)
    with torch.no_grad():
        pred_mask_low = seg_model(input_seg[None]).squeeze().argmax(0).cpu().numpy()
    
    mask_resizer = mT.Resize(half_dcm.shape, mode='nearest')
    pred_mask = mask_resizer(pred_mask_low[None]).squeeze().numpy()

    set1 = np.stack(np.where(pred_mask == 1), axis=0).T # Canal
    set2 = np.stack(np.where(pred_mask == 2), axis=0).T # Molar

    if len(set1) == 0 or len(set2) == 0:
        print("[!] Failure: Canal or Molar not detected.")
        return

    min_p1, min_p2, _ = find_min_distance_points_kdtree(set1, set2)
    center = ((min_p1 + min_p2) / 2).astype(int)
    cz, cy, cx = center
    m = 32 # Margin for 64x64x64 ROI

    roi_dcm = half_dcm[cz-m:cz+m, cy-m:cy+m, cx-m:cx+m]

    if roi_dcm.shape != (64, 64, 64):
        print(f"[!] Warning: ROI shape {roi_dcm.shape} is not 64x64x64. Resizing...")
        roi_resizer = mT.Resize([64, 64, 64])
        roi_dcm = roi_resizer(roi_dcm[None]).squeeze().numpy()
        roi_mask = roi_resizer(pred_mask[None], mode='nearest').squeeze().numpy()

    roi_dcm[roi_dcm > 3071] = 3071
    roi_dcm /= 3071.
    roi_sdm = signed_distance_map(roi_mask)

    input_class = torch.from_numpy(np.stack([roi_dcm, roi_sdm])).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = class_model(input_class)
        pred_idx = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    print("\n" + "="*30)
    print(f"  FINAL CLASSIFICATION RESULT")
    print("="*30)
    print(f"Predicted Hasegawa Type: {pred_idx + 1}")
    print(f"Confidence: {probs[pred_idx]*100:.2f}%")
    print(f"Full Probabilities: {np.round(probs, 4)}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Inference for M3-MC Relationship")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the sample CBCT DICOM directory")
    args = parser.parse_args()
    
    main(args)
