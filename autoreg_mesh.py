from utils import *

vtk.vtkOutputWindow.SetGlobalWarningDisplay(0)


def delete_mesh(mesh):
    # 3. 计算曲率
    curvature = vtk.vtkCurvatures()
    curvature.SetInputData(mesh)
    curvature.SetCurvatureTypeToMean()
    curvature.Update()

    # 4. 计算平均曲率
    curvature_array = curvature.GetOutput().GetPointData().GetArray("Mean_Curvature")
    num_points = curvature.GetOutput().GetNumberOfPoints()
    total_curvature = 0.0
    for i in range(num_points):
        total_curvature += curvature_array.GetValue(i)
    average_curvature = total_curvature / num_points

    # 5. 删除曲率低于平均值的三角面片
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(mesh)
    triangle_filter.Update()
    triangles = triangle_filter.GetOutput()

    for i in range(triangles.GetNumberOfCells()):
        cell = triangles.GetCell(i)
        point_ids = cell.GetPointIds()
        curvature_sum = 0.0
        for j in range(3):  # 遍历面片的三个顶点
            point_id = point_ids.GetId(j)
            curvature_sum += curvature_array.GetValue(point_id)
        if curvature_sum / 3.0 > average_curvature:
            triangles.DeleteCell(i)

    # 移除已删除的单元
    triangles.RemoveDeletedCells()

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(r'C:\daily_utils\img/朱宝莉仓扫-LowerJawScan2.stl')
    writer.SetInputData(triangles)
    writer.Write()

    return triangles

def convert_to_o3d_points(mesh):
    points = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(points)
    return source_pcd

source_mesh_file = r'C:\daily_utils\img/朱宝莉kou扫-LowerJawScan2.stl'  # 动
target_mesh_file = r'C:\daily_utils\img/朱宝莉仓扫-LowerJawScan2.stl'  # 不动

source_mesh_file2 = r'C:\daily_utils\img\2023自由手88例带种植体\0/1.stl'  # 动
target_mesh_file2 = r'C:\daily_utils\img/朱宝莉仓扫-LowerJawScan2.stl'  # 不动

target_mesh, _ = read_vtp_or_stl_file(target_mesh_file)
target_mesh2, _ = read_vtp_or_stl_file(target_mesh_file2)
# t_d_mesh = delete_mesh(target_mesh)
source_mesh, _ = read_vtp_or_stl_file(source_mesh_file)
source_mesh2, _ = read_vtp_or_stl_file(source_mesh_file2)
# s_d_mesh = delete_mesh(source_mesh)

source, target, source_down, target_down, trans_matrix = registration_polydata(
    source_mesh, target_mesh)

target2 = convert_to_o3d_points(target_mesh2)
source2 = convert_to_o3d_points(source_mesh2)
#
# source_temp = copy.deepcopy(source)
# target_temp = copy.deepcopy(target)
# source_temp.paint_uniform_color([1, 0.706, 0])
# target_temp.paint_uniform_color([0, 0.651, 0.929])
#
# o3d.visualization.draw_geometries([source_temp, target_temp])

draw_registration_result(source2, target2, trans_matrix)
