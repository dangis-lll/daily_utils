import os
import SimpleITK as sitk
from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    labelpath = r'C:\DL_DataBase\CBCT_data\skull\label/01-010-WTME.nii.gz'

    label_itk = sitk.ReadImage(labelpath)

    label_array = sitk.GetArrayFromImage(label_itk).astype('uint8')

    label_array[label_array <=4] = 0
    label_array[label_array > 4] = 1

    max_label = np.max(label_array,axis=0)

    count = np.sum(label_array, axis=(1, 2))
    # 找到1的个数最多的层
    max_index = np.argmax(count)
    max_count = count[max_index]
    print("1的个数最多的层是第{}层，共有{}个1".format(max_index, max_count))

    maxlayer = label_array[max_index]

    kernel = morphology.disk(5)
    img = morphology.erosion(maxlayer, kernel)
    kernel = morphology.disk(40)
    img = morphology.dilation(img, kernel)
    sk = morphology.skeletonize(img)


    maxlayer[sk==1]=2

    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))
    # ax[0].imshow(maxlayer, cmap='gray')  # 显示1最多的那层
    # ax[0].set_title('Max count layer')
    # ax[1].imshow(max_label, cmap='gray')
    # ax[1].set_title('Project layer')
    # ax[2].imshow(img, cmap='gray')
    # ax[3].set_title('ConvexHull layer')
    # ax[4].imshow(sk, cmap='gray')
    # ax[5].set_title('skeleton layer')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.imshow(sk, cmap='gray')
    ax.set_title('skeleton layer')
    plt.show()

    # plt.imshow(maxlayer, cmap='gray')
    # plt.show()






