import os

from utils import *

data_path = r'C:\baidunetdiskdownload\ddd\img'
bone_path = r'C:\baidunetdiskdownload\ddd\bone'
label_path = r'C:\baidunetdiskdownload\ddd\label'
out_path = r'C:\baidunetdiskdownload\ddd\train'

for file in os.listdir(data_path):
    ct, prop = read_nii_2_np(os.path.join(data_path, file))
    bone, _ = read_nii_2_np(os.path.join(bone_path, file))
    label, _ = read_nii_2_np(os.path.join(label_path, file))

    ct[np.isnan(ct)] = 0
    ct = ct.astype('float32')
    ct = ct - ct.min()
    # peak_id = get_peak_idx(ct)
    # ct[ct < peak_id] = 0
    ori_data_shape = ct.shape

    spacing = prop[0]
    mask = bone == 2
    roi_bbox = get_bbox_from_mask(mask)
    delta_mm = 3
    delta_dims = [int(delta_mm / spacing[2]), int(delta_mm / spacing[1]), int(delta_mm / spacing[0])]  # z,y,x
    for i in range(3):
        roi_bbox[i][0] = max(0, roi_bbox[i][0] - delta_dims[i])
        roi_bbox[i][1] = min(roi_bbox[i][1] + delta_dims[i], ori_data_shape[i])

    ct = crop_to_bbox(ct,roi_bbox)
    label = crop_to_bbox(label,roi_bbox)

    save_nii(ct,prop,os.path.join(out_path,'img',file))
    save_nii(label,prop,os.path.join(out_path,'label',file))
