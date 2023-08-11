import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy


# Read STL file
def read_vtp_or_stl_file(file_name):
    if file_name.endswith('.vtp'):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(file_name)
        reader.Update()
        mesh = reader.GetOutput()
        cell_data = mesh.GetCellData()
        vtk_data_array = cell_data.GetArray('Label')
        label = vtk_to_numpy(vtk_data_array)
        return mesh, label
    elif file_name.endswith('.stl'):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_name)
        reader.Update()
        mesh = reader.GetOutput()
        label = []
        return mesh, label
    else:
        raise ValueError("File extension must be .vtp or .stl")


# Compute the edges of a mesh
def compute_edges(mesh):
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(mesh)
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.NonManifoldEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.Update()

    return feature_edges.GetOutput()


# Create a VTK renderer
def create_renderer():
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.1, 0.1)
    return renderer


# Add mesh actor to the renderer
def add_mesh_actor(renderer, mesh, color):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)

    renderer.AddActor(actor)


# Display the mesh and its edges
def display_mesh_and_edges(mesh, edges):
    renderer = create_renderer()
    add_mesh_actor(renderer, mesh, (0.7, 0.7, 0.7))  # Original mesh in light gray color
    add_mesh_actor(renderer, edges, (1.0, 1.0, 1.0))  # Edges in red color

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    interactor.Initialize()
    render_window.Render()
    interactor.Start()


def extract_largest_component(polydata):
    connectivity_filter = vtk.vtkConnectivityFilter()
    connectivity_filter.SetInputData(polydata)
    connectivity_filter.SetExtractionModeToLargestRegion()
    connectivity_filter.Update()
    return connectivity_filter.GetOutput()


def extract_subset(mesh, tooth_id):
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(mesh)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, 'Label')
    threshold.SetLowerThreshold(tooth_id)
    threshold.SetUpperThreshold(tooth_id)
    threshold.Update()

    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(threshold.GetOutput())
    geometry_filter.Update()

    return extract_largest_component(geometry_filter.GetOutput())


def find_closest_vertices(edges, target_mesh):
    # Create a vtkPointLocator and set the target_mesh as input
    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(target_mesh)
    point_locator.BuildLocator()

    # Initialize an empty list to store the closest vertices

    # Iterate over all edge points
    closest_vertices = set()
    points = edges.GetPoints()
    for i in range(points.GetNumberOfPoints()):
        point = points.GetPoint(i)

        # Find the closest vertex in the target_mesh
        closest_vertex_id = point_locator.FindClosestPoint(point)

        closest_vertices.add(tuple(target_mesh.GetPoint(closest_vertex_id)))

    unique_closest_vertices = list(closest_vertices)
    closest_vertices = vtk.vtkPoints()
    for i in unique_closest_vertices:
        closest_vertices.InsertNextPoint(i)

    return closest_vertices


if __name__ == '__main__':

    ori_datapath = r"C:\DL_DataBase\IOS_data\ori_vtp/01-001-ZYHA-MaxillaryAnatomy_predicted.vtp"
    dec_datapath = r"C:\DL_DataBase\IOS_data\decimate_mesh/01-001-ZYHA-MaxillaryAnatomy.stl"

    # Read the STL file
    mesh, label = read_vtp_or_stl_file(ori_datapath)

    mesh_d, _ = read_vtp_or_stl_file(dec_datapath)
    n_cell_d = mesh_d.GetNumberOfCells()
    #
    label_init = vtk.vtkFloatArray()
    mesh_d.GetCellData().SetScalars(label_init)
    label_init.SetNumberOfTuples(n_cell_d)
    label_init.SetNumberOfComponents(1)
    for i in range(n_cell_d):
        label_init.SetComponent(i, 0, 0)

    mesh_d.GetCellData().GetAbstractArray(0).SetName("Label")

    tooth_id = np.unique(label)[1:]

    for id in tooth_id:
        mesh1 = extract_subset(mesh, id)
        # Compute the edges of the mesh
        edges = compute_edges(mesh1)

        edge_points = find_closest_vertices(edges, mesh_d)

        selectPolyData = vtk.vtkSelectPolyData()
        selectPolyData.GenerateSelectionScalarsOn()
        selectPolyData.GenerateUnselectedOutputOff()
        selectPolyData.SetEdgeSearchModeToDijkstra()
        selectPolyData.SetSelectionModeToSmallestRegion()

        selectPolyData.SetInputData(mesh_d)
        selectPolyData.SetLoop(edges.GetPoints())

        selectPolyData.Update()
        display_mesh_and_edges(selectPolyData.GetOutput(), edges)

        PtsId = vtk.vtkIdList()
        for j in range(n_cell_d):
            selectPolyData.GetOutput().GetCellPoints(j, PtsId)
            if selectPolyData.GetOutput().GetPointData().GetScalars().GetComponent(PtsId.GetId(0), 0) < .1 and \
                    selectPolyData.GetOutput().GetPointData().GetScalars().GetComponent(PtsId.GetId(1), 0) < .1 and \
                    selectPolyData.GetOutput().GetPointData().GetScalars().GetComponent(PtsId.GetId(2), 0) < .1:
                mesh_d.GetCellData().GetArray("Label").SetComponent(j, 0, id)

        mesh_d.GetCellData().GetArray("Label").Modified()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(r"C:\DL_DataBase\IOS_data\decimate_mesh/01-001-ZYHA-MaxillaryAnatomy.vtp")
    writer.SetInputData(mesh_d)
    writer.Write()

