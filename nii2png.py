import matplotlib

matplotlib.use('TkAgg')
import nibabel as nib
import numpy as np

import os
import imageio


# ---------------------------------------------#
# nii_path : nii文件的路径
# img_save_path : 切片的保存路径
# axis : 说明是沿着哪个方向切片的
# ---------------------------------------------
def nii_to_png(nii_path, img_save_path, axis):
    # 若保存路径不存在，则创建
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    nii = nib.load(nii_path)
    nii_fdata = nii.get_fdata()
    nii_fdata = np.rot90(nii_fdata)

    # 以切片的轴向作为保存png的子文件夹名
    foldername = axis
    png_save_path = os.path.join(img_save_path, foldername)
    if not os.path.exists(png_save_path):
        os.mkdir(png_save_path)

    flag = 100
    if axis == 'x':
        (axis, y, z) = nii.shape
        flag = 0
    elif axis == 'y':
        (x, axis, z) = nii.shape
        flag = 1
    elif axis == 'z':
        (x, y, axis) = nii.shape
        flag = 2
    else:
        print("wrong axis")

    for i in range(axis):
        if flag == 0:
            slice = nii_fdata[i, :, :]
        elif flag == 1:
            slice = nii_fdata[:, i, :]
        elif flag == 2:
            slice = nii_fdata[:, :, i]
        # 以数字1,2,3...为png图片命名
        imageio.imwrite(os.path.join(png_save_path, 'label_{}.png'.format(i)), slice)
def all_nii_to_png(all_nii_path,all_image_save_path, axis):
    all_nii_path_list = os.listdir(all_nii_path)
    for i in range(len(all_nii_path_list)):
        nii_to_png(os.path.join(all_nii_path,all_nii_path_list[i]),os.path.join(all_image_save_path, all_nii_path_list[i]), axis)
        print("第{}个nii文件转换完成！".format(i))

if __name__ == "__main__":
    all_nii_path = r'C:\DL_DataBase\CBCT_data\raw_data\test/nii'
    all_image_save_path = r'C:\DL_DataBase\CBCT_data\raw_data\test/png'
    all_nii_to_png(all_nii_path, all_image_save_path, 'z')

