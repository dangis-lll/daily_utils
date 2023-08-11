from scipy import ndimage
from scipy.ndimage import morphology

from utils import *
import SimpleITK as sitk
import os

import numpy as np
if __name__ == '__main__':
    filepath = r'C:\DL_DataBase\CBCT_data\alltooth\label'
    datalist = os.listdir(filepath)
    for file in datalist:
        datapath = os.path.join(filepath,file)
        ct_array,prop = read_nii_2_np(datapath)
        labels = np.unique(ct_array)
        labels=labels[1:]
        new_label = np.zeros_like(ct_array).astype('float32')
        for i in labels:
            label = np.zeros_like(ct_array)
            label[ct_array==i]=1
            keep_connected_regions(label, label, [1])

            distance_image = ndimage.distance_transform_edt(label)
            distance_image = (distance_image-distance_image.min())/(distance_image.max()-distance_image.min())
            new_label+=distance_image

        save_nii(new_label,prop,os.path.join(r'C:\DL_DataBase\CBCT_data\alltooth\dismap',file))





    # toothsize = load_json('toothsize.json')
    # T = []
    # for k,v in toothsize.items():
    #     for id,size in v.items():
    #         if id =="2" or id == "17":
    #             continue
    #         T.append(size)
    # X = np.array(T)[:,2]
    # Y = np.array(T)[:,1]
    # Z = np.array(T)[:,0]
    # print(np.max(np.clip(X,np.percentile(X,0.05),np.percentile(X,99.5))))
    # print(np.max(np.clip(Y,np.percentile(Y,0.05),np.percentile(Y,99.5))))
    # print(np.max(np.clip(Z,np.percentile(Z,0.05),np.percentile(Z,99.5))))



