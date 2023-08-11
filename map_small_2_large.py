from scipy.ndimage import morphology

from utils import *
import SimpleITK as sitk
import os
from scipy.signal import correlate
import numpy as np

if __name__ == '__main__':
    oriimg_path = r'C:\DL_DataBase\CBCT_data\CTooth\NC release data\img'
    roiimg_path = r'C:\DL_DataBase\CBCT_data\kk\img'
    outputpath = r'C:\DL_DataBase\CBCT_data\CTooth\NC release data\new'
    datalist = os.listdir(roiimg_path)
    for file in datalist:
        if os.path.exists(os.path.join(outputpath,file)):
            continue
        print(file)
        ct_array, prop = read_nii_2_np(os.path.join(oriimg_path, file))
        label_array, _ = read_nii_2_np(os.path.join(roiimg_path.replace('img', 'label'), file))
        tooth_array, _ = read_nii_2_np(os.path.join(roiimg_path, file))
        ct_array = ct_array.astype('float32')
        tooth_array = tooth_array.astype('float32')
        new_label = np.zeros_like(ct_array).astype('uint8')


        img_size = ct_array.shape
        tooth_size = tooth_array.shape

        result = correlate(ct_array, tooth_array, mode='valid')
        # 找到结果中的最大值的位置
        max_loc = np.unravel_index(np.argmax(result), result.shape)

        new_label[max_loc[0]:max_loc[0]+tooth_size[0],max_loc[1]:max_loc[1]+tooth_size[1],max_loc[2]:max_loc[2]+tooth_size[2]] = label_array

        save_nii(new_label,prop,os.path.join(outputpath,file))

