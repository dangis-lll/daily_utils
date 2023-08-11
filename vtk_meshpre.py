import vtk
from utils import *
import os

if __name__ == '__main__':
    mesh, label = read_vtp_or_stl_file(r'C:\DL_code\ScanSeg\Dataset\raw\up\01-001-ZYHA-MaxillaryAnatomy_predicted.vtp')

    num_cells = mesh.GetNumberOfCells()

    curvatures_filter = vtk.vtkCurvatures()
    curvatures_filter.SetInputData(mesh)
    # 选择曲率类型（高斯曲率或平均曲率）
    curvatures_filter.SetCurvatureTypeToGaussian()  # 或者使用 curvatures_filter.SetCurvatureTypeToMean()
    # 更新过滤器以计算曲率
    curvatures_filter.Update()
    # 获取带有曲率值的PolyData输出
    output_poly_data = curvatures_filter.GetOutput()
    point_curvatures = output_poly_data.GetPointData().GetScalars()
    curvatures_array = np.zeros([num_cells, 3], dtype='float32')
    for i in range(num_cells):
        curvatures_array[i][0] = point_curvatures.GetValue(output_poly_data.GetCell(i).GetPointId(0))
        curvatures_array[i][1] = point_curvatures.GetValue(output_poly_data.GetCell(i).GetPointId(1))
        curvatures_array[i][2] = point_curvatures.GetValue(output_poly_data.GetCell(i).GetPointId(2))

    cells = np.zeros([num_cells, 9], dtype='float32')
    for i in range(len(cells)):
        cells[i][0], cells[i][1], cells[i][2] = mesh.GetPoint(
            mesh.GetCell(i).GetPointId(0))  # don't need to copy
        cells[i][3], cells[i][4], cells[i][5] = mesh.GetPoint(
            mesh.GetCell(i).GetPointId(1))  # don't need to copy
        cells[i][6], cells[i][7], cells[i][8] = mesh.GetPoint(
            mesh.GetCell(i).GetPointId(2))  # don't need to copy

    cmf = vtk.vtkCenterOfMass()
    cmf.SetInputData(mesh)
    cmf.Update()
    c = cmf.GetCenter()
    mean_cell_centers = np.array(c)

    cells[:, 0:3] -= mean_cell_centers[0:3]
    cells[:, 3:6] -= mean_cell_centers[0:3]
    cells[:, 6:9] -= mean_cell_centers[0:3]

    pdn = vtk.vtkPolyDataNormals()
    pdn.SetInputData(mesh)
    pdn.SetComputeCellNormals(1)
    pdn.SetSplitting(0)
    pdn.Update()
    normals = np.zeros([num_cells, 3], dtype='float32')
    for i in range(num_cells):
        normals[i][0] = pdn.GetOutput().GetCellData().GetNormals().GetTuple(i)[0]
        normals[i][1] = pdn.GetOutput().GetCellData().GetNormals().GetTuple(i)[1]
        normals[i][2] = pdn.GetOutput().GetCellData().GetNormals().GetTuple(i)[2]


    vcen = vtk.vtkCellCenters()
    vcen.SetInputData(mesh)
    vcen.Update()
    barycenters = vtk_to_numpy(vcen.GetOutput().GetPoints().GetData())

    barycenters -= mean_cell_centers[0:3]

    v1 = np.zeros([num_cells, 3], dtype='float32')
    v2 = np.zeros([num_cells, 3], dtype='float32')
    v3 = np.zeros([num_cells, 3], dtype='float32')

    v1[:, 0] = cells[:, 0] - cells[:, 3]
    v1[:, 1] = cells[:, 1] - cells[:, 4]
    v1[:, 2] = cells[:, 2] - cells[:, 5]

    v2[:, 0] = cells[:, 3] - cells[:, 6]
    v2[:, 1] = cells[:, 4] - cells[:, 7]
    v2[:, 2] = cells[:, 5] - cells[:, 8]

    v3[:, 0] = cells[:, 6] - cells[:, 0]
    v3[:, 1] = cells[:, 7] - cells[:, 1]
    v3[:, 2] = cells[:, 8] - cells[:, 2]

    mesh_normals1 = np.cross(v3, v1)  # calculating the normal for point P1
    mesh_normal_length1 = np.linalg.norm(mesh_normals1, axis=1)
    mesh_normals1[:, 0] /= mesh_normal_length1[:]
    mesh_normals1[:, 1] /= mesh_normal_length1[:]
    mesh_normals1[:, 2] /= mesh_normal_length1[:]

    mesh_normals2 = np.cross(v1, v2)  # calculating the normal for point P2
    mesh_normal_length2 = np.linalg.norm(mesh_normals2, axis=1)
    mesh_normals2[:, 0] /= mesh_normal_length2[:]
    mesh_normals2[:, 1] /= mesh_normal_length2[:]
    mesh_normals2[:, 2] /= mesh_normal_length2[:]

    mesh_normals3 = np.cross(v2, v3)  # calculating the normal for point P3
    mesh_normal_length3 = np.linalg.norm(mesh_normals3, axis=1)
    mesh_normals3[:, 0] /= mesh_normal_length3[:]
    mesh_normals3[:, 1] /= mesh_normal_length3[:]
    mesh_normals3[:, 2] /= mesh_normal_length3[:]

    points = vtk_to_numpy(mesh.GetPoints().GetData())

    maxs = points.max(axis=0)
    mins = points.min(axis=0)
    nmeans = normals.mean(axis=0)
    nstds = normals.std(axis=0)
    means = points.mean(axis=0)
    stds = points.std(axis=0)
    cmeans = curvatures_array.mean(axis=0)
    cstds = curvatures_array.std(axis=0)

    nmeans1 = mesh_normals1.mean(axis=0)
    nstds1 = mesh_normals1.std(axis=0)
    nmeans2 = mesh_normals2.mean(axis=0)
    nstds2 = mesh_normals2.std(axis=0)
    nmeans3 = mesh_normals3.mean(axis=0)
    nstds3 = mesh_normals3.std(axis=0)

    for i in range(3):
        cells[:, i] = (cells[:, i] - means[i]) / stds[i]  # point 1
        cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]  # point 2
        cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]  # point 3
        normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]
        barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
        mesh_normals1[:, i] = (mesh_normals1[:, i] - nmeans1[i]) / nstds1[i]
        mesh_normals2[:, i] = (mesh_normals2[:, i] - nmeans2[i]) / nstds2[i]
        mesh_normals3[:, i] = (mesh_normals3[:, i] - nmeans3[i]) / nstds3[i]
        curvatures_array[:, i] = (curvatures_array[:, i] - cmeans[i]) / cstds[i]

    X = np.column_stack((cells, barycenters, normals,mesh_normals1,mesh_normals2,mesh_normals2,curvatures_array))
    print(X.shape)

