import pickle
import numpy as np
import SimpleITK as sitk
import torch
import os
from utils import *

if __name__ == '__main__':

    labelpath = r'C:\DL_DataBase\CBCT_data\raw_data\ianseg\label'
    imgpath = r'C:\DL_DataBase\CBCT_data\raw_data\ianseg/img'
    outpath = r'C:\DL_DataBase\CBCT_data\raw_data\ianseg\ian_train'
    datalist = os.listdir(imgpath)
    for data in datalist:
        ct_array, prop = read_nii_2_np(os.path.join(imgpath,data))
        label_array, _ = read_nii_2_np(os.path.join(labelpath,data))

        z = ct_array.shape[2]
        ct_l = ct_array[:, :, :z // 2]
        ct_r = ct_array[:, :, z // 2:]
        label_l = label_array[:, :, :z // 2]
        label_r = label_array[:, :, z // 2:]
        filename = data[:-7]
        save_nii(ct_l, prop, os.path.join(outpath, 'img/{}_l.nii.gz'.format(filename)))
        save_nii(ct_r, prop, os.path.join(outpath, 'img/{}_r.nii.gz'.format(filename)))
        save_nii(label_l,prop,os.path.join(outpath, 'label/{}_l.nii.gz'.format(filename)))
        save_nii(label_r,prop,os.path.join(outpath, 'label/{}_r.nii.gz'.format(filename)))
