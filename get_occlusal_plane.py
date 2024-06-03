import matplotlib.pyplot as plt
from utils import *


def project_to_plane(point, a, b, c):
    d = -(a * point[0] + b * point[1] - point[2] + c) / (a * a + b * b + 1)
    projected_point = point + d * np.array([a, b, -1])
    return projected_point


label, prop = read_nii_2_np(r'C:\DL_DataBase\CBCT_data\skull\tooth/BTLUb.nii.gz')
# label, prop = read_nii_2_np(r'C:\Users\dangis\Desktop\paper\CBCT_paper\figures\tooth/206.nii.gz')
u_points = []
l_points = []
points = []

sample_points= []

ids = np.unique(label)[1:]

# ids = [8,7,6,5,4,3,2,1,9,10,11,12,13,14,15,16,32,31,30,29,28,27,26,25,17,18,19,20,21,22,23,24]

for i in ids:
    tooth = label == i
    indices = np.where(tooth == 1)
    center_x = np.mean(indices[0])
    center_y = np.mean(indices[1])
    center_z = np.mean(indices[2])
    points.append([center_z, center_y, center_x])
    if i < 17:
        u_points.append([center_z, center_y, center_x])
    else:
        l_points.append([center_z, center_y, center_x])

# 生成一些示例的空间点
u_points = np.array(u_points)
l_points = np.array(l_points)
points = np.array(points)
#
# 使用最小二乘法拟合平面
# 构造增广矩阵
u_A = np.c_[u_points[:, 0], u_points[:, 1], np.ones(u_points.shape[0])]
u_b = u_points[:, 2]

# 最小二乘法解方程 Ax = b
u_x, u_residuals, u_rank, u_singular_values = np.linalg.lstsq(u_A, u_b, rcond=None)

# 提取平面参数
u_a, u_b, u_c = u_x

l_A = np.c_[l_points[:, 0], l_points[:, 1], np.ones(l_points.shape[0])]
l_b = l_points[:, 2]

# 最小二乘法解方程 Ax = b
l_x, l_residuals, l_rank, l_singular_values = np.linalg.lstsq(l_A, l_b, rcond=None)

# 提取平面参数
l_a, l_b, l_c = l_x

a_avg = (u_a + l_a) / 2
b_avg = (u_b + l_b) / 2
c_avg = (u_c + l_c) / 2

# 投影所有点到平面上
projected_points = np.array([project_to_plane(point, a_avg, b_avg, c_avg) for point in points])

# 绘制原始空间点
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# 绘制拟合的平面
x_range = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 10)
y_range = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 10)
X, Y = np.meshgrid(x_range, y_range)
Z = a_avg * X + b_avg * Y + c_avg
ax.plot_surface(X, Y, Z, alpha=0.5)

ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
ax.scatter(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2], c='r', marker='o')

# 设置图形属性
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Fitted Plane')

plt.show()
