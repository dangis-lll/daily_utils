import os
from monai.transforms import Resample
from utils import *

bpath = r'C:\DL_DataBase\CBCT_data\femur'
outpath = r'C:\DL_DataBase\CBCT_data\femur\train'

for i in os.listdir(os.path.join(bpath,'img')):
    if os.path.exists(os.path.join(outpath,'img',i)):
        continue
    ct,prop = read_nii_2_np(os.path.join(bpath,'img',i))
    label,_ = read_nii_2_np(os.path.join(bpath,'label',i))
    target_spacing = [1,1,1]
    origin_spacing = [prop[0][2],prop[0][1],prop[0][0]]
    s = ct.shape
    new_shape = np.round(
        ((np.array(origin_spacing) / np.array(target_spacing)).astype(float) * np.array(
            ct.shape))).astype(int)
    ct = resample_data2(ct,new_shape)
    label = resample_data2(label,new_shape)

    mask = label!=0

    roi_bbox = get_bbox_from_mask(mask)
    delta_mm = 10
    delta_dims = [int(delta_mm / target_spacing[0]), int(delta_mm / target_spacing[1]), int(delta_mm / target_spacing[2])]  # z,y,x
    for j in range(3):
        roi_bbox[j][0] = max(0, roi_bbox[j][0] - delta_dims[j])
        roi_bbox[j][1] = min(roi_bbox[j][1] + delta_dims[j], ct.shape[j])

    ct = crop_to_bbox(ct, roi_bbox)
    label = crop_to_bbox(label, roi_bbox)

    # ct[ct<0]=0
    ct = ct-ct.min()-1000
    ct[ct < 0] = 0

    prop[0] = target_spacing
    save_nii(ct,prop,os.path.join(outpath,'img',i))
    save_nii(label,prop,os.path.join(outpath,'label',i))
