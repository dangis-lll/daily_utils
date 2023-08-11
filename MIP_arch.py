import os
import SimpleITK as sitk
from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    imgpath = r'C:\DL_DataBase\CBCT_data\raw_data\img/01-001-ZYHA.nii.gz'

    img_itk = sitk.ReadImage(imgpath)

    img_array = sitk.GetArrayFromImage(img_itk)

    max_label = np.max(img_array, axis=2)

    mean = np.mean(max_label)
    std = np.std(max_label)

    # max_label  = (max_label-mean)/std
    #
    # max_label[max_label>1.5*mean]=max_label.max()
    # high_pixels = np.argwhere(max_label > mean)
    #
    # # 找到灰度值相对较高的一部分
    # y_min, x_min = high_pixels.min(axis=0)
    # y_max, x_max = high_pixels.max(axis=0)
    # high_part = max_label[y_min:y_max + 1, x_min:x_max + 1]

    plt.imshow(max_label, cmap='gray')
    plt.show()

