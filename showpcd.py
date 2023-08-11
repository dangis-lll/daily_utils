import numpy as np
import open3d as o3d

# 生成随机的点云和颜色

if __name__ == '__main__':
    array = np.load(r'C:\DL_DataBase\CBCT_data\raw_data\nii_pcd\01-027-LQYU_max.npz')['img']
    # array = array[np.random.choice(len(array), 20000, replace=False)]
    point_cloud = array[:,:3]
    # label = array[:,3:]
    # label = label/label.max()
    # colors = np.column_stack((label,label,label))


    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    # 绘制点云
    o3d.visualization.draw_geometries([pcd])
    #
