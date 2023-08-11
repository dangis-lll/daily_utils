import os
import time
import numpy as np
from utils import *

#
# labelpath = r'C:\DL_DataBase\CBCT_data\alltooth\fine_data\re\label'
# labellist = os.listdir(labelpath)
#
# ave_label = []
#
# for i in labellist:
#     label_array, prop = read_nii_2_np(os.path.join(labelpath, i))
#
#     labels = np.unique(label_array)
#     new_label = [0] * 33
#     for id in labels:
#         num_id = np.sum(label_array==id)
#         new_label[id]=num_id
#     ave_label.append(new_label)
#
# np.save('tooth_sum',np.array(ave_label))

new_label = np.load('tooth_sum.npy')
ave_label = np.vstack(new_label)
# ave_label = ave_label[:,1:]

column_means = np.zeros(ave_label.shape[1])

# 遍历每一列
for col_idx in range(ave_label.shape[1]):
    # 提取当前列
    column = ave_label[:, col_idx]

    # 创建一个布尔数组，表示列中的非零元素
    non_zero_mask = column != 0

    # 计算非零元素的平均值并将其存储在 column_means 中
    column_means[col_idx] = np.mean(column[non_zero_mask])

print("每列忽略0的平均值:", column_means)
column_means = np.round(column_means)


pixel_frequency = np.array(column_means) / np.sum(column_means)
pixel_frequency = pixel_frequency.astype('float32')
# 计算每个类别的权重（归一化倒数）
class_weights = 1.0 / pixel_frequency
class_weights /= class_weights.sum()
class_weights *=100
class_weights[1:] /= class_weights[1:].min()
np.save('my_weights',class_weights)
print(class_weights)
