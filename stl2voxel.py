import os

import numpy as np
from madcad.hashing import PositionMap
from madcad.io import read
from utils import *

mesh_path = r'C:\DL_DataBase\IOS_data\ori_data'
output_path = r'C:\DL_DataBase\IOS_data\voxel'
for file in os.listdir(mesh_path):
    spacing = 0.25
    mesh = read(os.path.join(mesh_path,file))

    voxel = set()  # this is a sparse voxel
    hasher = PositionMap(spacing)  # ugly object creation, just to use one of its methods
    for face in mesh.faces:
        voxel.update(hasher.keysfor(mesh.facepoints(face)))
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # # 将坐标分解为三个列表
    # x, y, z = zip(*voxel)
    #
    # # 创建一个3D坐标图
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # 绘制散点图
    # ax.scatter(x, y, z, c='r', marker='o')
    #
    # # 设置坐标轴标签
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # # 显示图形
    # plt.show()
    voxel = np.array(list(voxel))
    coordinates = voxel - voxel.min(axis=0)
    image_size = np.ceil(coordinates.max(axis=0)).astype(int) + 1
    image = np.zeros(image_size)
    # 将坐标位置设置为1
    image[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = 1
    image = np.transpose(image, (2, 1, 0))

    ct_itk = sitk.GetImageFromArray(image)
    ct_itk.SetSpacing([spacing, spacing, spacing])
    save_name = file[:-4]+'.nii.gz'
    sitk.WriteImage(ct_itk, os.path.join(output_path,file[:-4]+'.nii.gz'))




