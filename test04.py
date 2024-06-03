import os

from utils import *


datalist = r'C:\DL_DataBase\CBCT_data\sinus\label'
bonelist = r'C:\DL_DataBase\CBCT_data\sinus\bone'

for i in os.listdir(datalist):
    label,p = read_nii_2_np(os.path.join(datalist,i))
    bone,_ = read_nii_2_np(os.path.join(bonelist,i))

    bone[bone==2]=0
    bone[label!=0]=2

    save_nii(bone,p,os.path.join(bonelist,i))