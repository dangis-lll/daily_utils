import os

import numpy as np

from utils import *

data_path = r'C:\DL_DataBase\bone\public_data'
output_path = r'C:\DL_DataBase\bone\crop'
data_list = os.listdir(os.path.join(data_path,'img'))

for file in data_list:
    img_data = os.path.join(data_path,'img',file)
    label_data = os.path.join(data_path,'label',file)
    ct,prop = read_nii_2_np(img_data)
    label,_ = read_nii_2_np(label_data)
    mask = label>0
    bbox = get_bbox_from_mask(mask)
    z_max = bbox[0][1]
    z_max = z_max+20
    if z_max >ct.shape[0]:
        z_max = ct.shape[0]
    bbox = [[0,z_max],[0,ct.shape[1]],[0,ct.shape[2]]]
    ct_array = crop_to_bbox(ct,bbox)
    label_array = crop_to_bbox(label,bbox)

    save_nii(ct_array,prop,os.path.join(output_path,'img',file))
    save_nii(label_array,prop,os.path.join(output_path,'label',file))



