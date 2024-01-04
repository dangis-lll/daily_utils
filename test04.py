import numpy as np
from scipy.ndimage import rotate, shift
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *

# 创建一个示例的三维CT图像（假设图像大小为30x30x30）
path = r'C:\DL_DataBase\CBCT_data\skull\img/01-001-ZYHA.nii.gz'
ct_image,prop = read_nii_2_np(path)

# 随机生成平移和旋转参数
translation = np.random.randint(-10, 10, size=3)  # 随机平移值
rotation_angles = np.random.randint(-30, 30, size=3)  # 随机旋转角度

# 进行平移
ct_image_shifted = shift(ct_image, translation, mode='constant', cval=0)

# 进行旋转
ct_image_rotated = rotate(ct_image_shifted, angle=rotation_angles, axes=(0, 1), mode='constant', cval=0)
ct_image_rotated = rotate(ct_image_rotated, angle=rotation_angles, axes=(0, 2), mode='constant', cval=0)
ct_image_rotated = rotate(ct_image_rotated, angle=rotation_angles, axes=(2, 1), mode='constant', cval=0)

save_nii(ct_image_rotated,prop,r'C:\DL_DataBase\CBCT_data\skull/01-001-ZYHA.nii.gz')

# # 可视化原始、平移和旋转后的图像
# fig = plt.figure(figsize=(10, 5))
#
# # 原始图像
# ax1 = fig.add_subplot(131, projection='3d')
# ax1.voxels(ct_image, edgecolor='k')
# ax1.set_title('Original Image')
#
# # 平移后的图像
# ax2 = fig.add_subplot(132, projection='3d')
# ax2.voxels(ct_image_shifted, edgecolor='k')
# ax2.set_title('Shifted Image')
#
# # 旋转后的图像
# ax3 = fig.add_subplot(133, projection='3d')
# ax3.voxels(ct_image_rotated, edgecolor='k')
# ax3.set_title('Rotated Image')
#
# plt.show()
