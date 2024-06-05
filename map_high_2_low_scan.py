# import numpy as np
# import vtk
# from vtkmodules.util.numpy_support import vtk_to_numpy
#
#
# # Read STL file
# def read_vtp_or_stl_file(file_name):
#     if file_name.endswith('.vtp'):
#         reader = vtk.vtkXMLPolyDataReader()
#         reader.SetFileName(file_name)
#         reader.Update()
#         mesh = reader.GetOutput()
#         cell_data = mesh.GetCellData()
#         vtk_data_array = cell_data.GetArray('Label')
#         label = vtk_to_numpy(vtk_data_array)
#         return mesh, label
#     elif file_name.endswith('.stl'):
#         reader = vtk.vtkSTLReader()
#         reader.SetFileName(file_name)
#         reader.Update()
#         mesh = reader.GetOutput()
#         label = []
#         return mesh, label
#     else:
#         raise ValueError("File extension must be .vtp or .stl")
#
#
# # Compute the edges of a mesh
# def compute_edges(mesh):
#     feature_edges = vtk.vtkFeatureEdges()
#     feature_edges.SetInputData(mesh)
#     feature_edges.BoundaryEdgesOn()
#     feature_edges.FeatureEdgesOff()
#     feature_edges.NonManifoldEdgesOff()
#     feature_edges.ManifoldEdgesOff()
#     feature_edges.Update()
#
#     return feature_edges.GetOutput()
#
#
# # Create a VTK renderer
# def create_renderer():
#     renderer = vtk.vtkRenderer()
#     renderer.SetBackground(0.1, 0.1, 0.1)
#     return renderer
#
#
# # Add mesh actor to the renderer
# def add_mesh_actor(renderer, mesh, color):
#     mapper = vtk.vtkPolyDataMapper()
#     mapper.SetInputData(mesh)
#
#     actor = vtk.vtkActor()
#     actor.SetMapper(mapper)
#     actor.GetProperty().SetColor(color)
#
#     renderer.AddActor(actor)
#
#
# # Display the mesh and its edges
# def display_mesh_and_edges(mesh, edges):
#     renderer = create_renderer()
#     add_mesh_actor(renderer, mesh, (0.7, 0.7, 0.7))  # Original mesh in light gray color
#     add_mesh_actor(renderer, edges, (1.0, 1.0, 1.0))  # Edges in red color
#
#     render_window = vtk.vtkRenderWindow()
#     render_window.AddRenderer(renderer)
#
#     interactor = vtk.vtkRenderWindowInteractor()
#     interactor.SetRenderWindow(render_window)
#
#     interactor.Initialize()
#     render_window.Render()
#     interactor.Start()
#
#
# def extract_largest_component(polydata):
#     connectivity_filter = vtk.vtkConnectivityFilter()
#     connectivity_filter.SetInputData(polydata)
#     connectivity_filter.SetExtractionModeToLargestRegion()
#     connectivity_filter.Update()
#     return connectivity_filter.GetOutput()
#
#
# def extract_subset(mesh, tooth_id):
#     threshold = vtk.vtkThreshold()
#     threshold.SetInputData(mesh)
#     threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, 'Label')
#     threshold.SetLowerThreshold(tooth_id)
#     threshold.SetUpperThreshold(tooth_id)
#     threshold.Update()
#
#     geometry_filter = vtk.vtkGeometryFilter()
#     geometry_filter.SetInputData(threshold.GetOutput())
#     geometry_filter.Update()
#
#     return extract_largest_component(geometry_filter.GetOutput())
#
#
# def find_closest_vertices(edges, target_mesh):
#     # Create a vtkPointLocator and set the target_mesh as input
#     point_locator = vtk.vtkPointLocator()
#     point_locator.SetDataSet(target_mesh)
#     point_locator.BuildLocator()
#
#     # Initialize an empty list to store the closest vertices
#
#     # Iterate over all edge points
#     closest_vertices = set()
#     points = edges.GetPoints()
#     for i in range(points.GetNumberOfPoints()):
#         point = points.GetPoint(i)
#
#         # Find the closest vertex in the target_mesh
#         closest_vertex_id = point_locator.FindClosestPoint(point)
#
#         closest_vertices.add(tuple(target_mesh.GetPoint(closest_vertex_id)))
#
#     unique_closest_vertices = list(closest_vertices)
#     closest_vertices = vtk.vtkPoints()
#     for i in unique_closest_vertices:
#         closest_vertices.InsertNextPoint(i)
#
#     return closest_vertices
#
#
# if __name__ == '__main__':
#
#     ori_datapath = r"C:\DL_DataBase\Scan_data/01-001-ZYHA-MandibularAnatomy_Normal_Bite.vtp"
#     dec_datapath = r"C:\DL_DataBase\Scan_data\mesh_d/01-001-ZYHA-MandibularAnatomy_Normal_Bite.stl"
#
#     # Read the STL file
#     mesh, label = read_vtp_or_stl_file(ori_datapath)
#
#     mesh_d, _ = read_vtp_or_stl_file(dec_datapath)
#     n_cell_d = mesh_d.GetNumberOfCells()
#     #
#     label_init = vtk.vtkFloatArray()
#     mesh_d.GetCellData().SetScalars(label_init)
#     label_init.SetNumberOfTuples(n_cell_d)
#     label_init.SetNumberOfComponents(1)
#     for i in range(n_cell_d):
#         label_init.SetComponent(i, 0, 0)
#
#     mesh_d.GetCellData().GetAbstractArray(0).SetName("Label")
#
#     tooth_id = np.unique(label)[1:]
#
#     for id in tooth_id:
#         mesh1 = extract_subset(mesh, id)
#         # Compute the edges of the mesh
#         edges = compute_edges(mesh1)
#
#         edge_points = find_closest_vertices(edges, mesh_d)
#
#         selectPolyData = vtk.vtkSelectPolyData()
#         selectPolyData.GenerateSelectionScalarsOn()
#         selectPolyData.GenerateUnselectedOutputOff()
#         selectPolyData.SetEdgeSearchModeToDijkstra()
#         selectPolyData.SetSelectionModeToSmallestRegion()
#
#         selectPolyData.SetInputData(mesh_d)
#         selectPolyData.SetLoop(edges.GetPoints())
#
#         selectPolyData.Update()
#         display_mesh_and_edges(selectPolyData.GetOutput(), edges)
#
#         PtsId = vtk.vtkIdList()
#         for j in range(n_cell_d):
#             selectPolyData.GetOutput().GetCellPoints(j, PtsId)
#             if selectPolyData.GetOutput().GetPointData().GetScalars().GetComponent(PtsId.GetId(0), 0) < .1 and \
#                     selectPolyData.GetOutput().GetPointData().GetScalars().GetComponent(PtsId.GetId(1), 0) < .1 and \
#                     selectPolyData.GetOutput().GetPointData().GetScalars().GetComponent(PtsId.GetId(2), 0) < .1:
#                 mesh_d.GetCellData().GetArray("Label").SetComponent(j, 0, id)
#
#         mesh_d.GetCellData().GetArray("Label").Modified()
#
#     writer = vtk.vtkXMLPolyDataWriter()
#     writer.SetFileName(r"C:\DL_DataBase\Scan_data/ddd.vtpmy.vtp")
#     writer.SetInputData(mesh_d)
#     writer.Write()
# 以上为使用原模型牙齿的边缘来获得简化模型的标注结果，存在问题无法解决，故使用下面的基于knn分类的方法进行处理
import os

import numpy as np

from utils import *

import faiss
import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy


class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions


def get_barycenters(mesh):
    vcen = vtk.vtkCellCenters()
    vcen.SetInputData(mesh)
    vcen.Update()
    barycenters = vtk_to_numpy(vcen.GetOutput().GetPoints().GetData())
    return barycenters


def save_meshseg(mesh, fine_labels, save_path):
    mylabels = vtk.vtkFloatArray()
    mesh.GetCellData().SetScalars(mylabels)
    mylabels.SetNumberOfTuples(fine_labels.shape[0])
    mylabels.SetNumberOfComponents(1)
    for i in range(fine_labels.shape[0]):
        mylabels.SetComponent(i, 0, fine_labels[i])
    mesh.GetCellData().GetAbstractArray(0).SetName("Label")
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(mesh)
    writer.SetFileName(save_path)
    writer.Update()
    writer.Write()


def upsampling_labels(input_mesh, refined_mesh_d, refine_labels_d):
    neigh = FaissKNeighbors(k=1)
    barycenters = get_barycenters(refined_mesh_d)
    ori_barycenters = get_barycenters(input_mesh)
    neigh.fit(barycenters, np.ravel(refine_labels_d))
    fine_labels = neigh.predict(ori_barycenters)
    fine_labels = fine_labels.reshape(-1, 1).astype(np.uint8)
    return fine_labels

ori_data_path = r''
dec_data_path = r''
out_data_path = r''
for i in os.listdir(ori_data_path):
    mesh, label = read_vtp_or_stl_file(os.path.join(ori_data_path, i))
    mesh_d, _ = read_vtp_or_stl_file(os.path.join(dec_data_path, i))
    label = label.astype('int64')
    label = upsampling_labels(mesh_d, mesh, label)

    save_meshseg(mesh_d, label, os.path.join(out_data_path, i))
