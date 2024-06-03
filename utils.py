import copy
import json
import pickle
from collections import OrderedDict

import monai.transforms
import numpy as np
import torch
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, medfilt
from skimage.transform import resize

import SimpleITK as sitk
import os
import vtk
from torch.nn.functional import grid_sample
from vtkmodules.util.numpy_support import vtk_to_numpy
from skimage import measure as meas
from vtkmodules.util import numpy_support
import open3d as o3d


def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    nonzero_mask = np.zeros(data.shape, dtype=bool)

    this_mask = data != 0
    nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    # resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    # image = image[resizer]
    image_ = image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]: bbox[2][1]]
    return image_


def get_peak_idx(ct_array):
    data_flat = ct_array.flatten()
    hist, _ = np.histogram(data_flat, bins=np.arange(np.min(data_flat), np.max(data_flat) + 1))
    hist_m = medfilt(hist, 5)
    hist_r = hist_m[::-1]
    peaks_indices, _ = find_peaks(hist_r, width=10, height=0.001 * np.sum(hist_m))
    peak_x = len(hist_r) - peaks_indices[0]
    return peak_x


def data_preprocess(ct_array, prop=None, target_spacing=None):
    ct_array[np.isnan(ct_array)] = 0
    ct_array = ct_array.astype('float32')
    ct_array = ct_array - ct_array.min()
    peak_x = get_peak_idx(ct_array)
    if target_spacing and prop:
        new_shape = np.round(
            ((np.array(prop[0]) / np.array(target_spacing)).astype(float) * np.array(
                ct_array.shape))).astype(int)
        ct_array = resample_data(ct_array, new_shape)
    ct_array = ct_array - peak_x
    ct_array[ct_array < 0] = 0
    # ct_array = np.clip(ct_array, ct_array.min(), np.percentile(ct_array, 99.5)).astype('float32')
    return ct_array


def resample_data(data, new_shape):
    assert len(data.shape) == 3, "data must be (z, x, y)"
    shape = (data.shape[0], data.shape[1], data.shape[2])
    new_shape = (new_shape[0], new_shape[1], new_shape[2])
    if shape != new_shape:
        dtype_data = data.dtype
        data = data.astype(float)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data[None][None]
        data = torch.tensor(data).to(device)
        reshaped_final_data = torch.nn.functional.interpolate(data, size=new_shape, mode='nearest').cpu().numpy()
        # print('new shape: ', reshaped_final_data[0][0].shape)
        return reshaped_final_data[0][0].astype(dtype_data)
    else:
        print("no resampling necessary")
        return data


def resample_data_or_seg_slow(data, new_shape, is_seg, order=3):
    assert len(data.shape) == 3, "data must be (z, x, y)"
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    data = data.astype(float)
    shape = np.array(data.shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        reshaped_final_data = resize_fn(data, new_shape, order, **kwargs)
        # seg:order=1
        # ct:order=3
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data.astype(dtype_data)


def resample_label(data, new_shape):
    assert len(data.shape) == 3, "data must be (z, x, y)"
    resize_fn = resize
    kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    data = data.astype(float)
    shape = np.array(data.shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        reshaped_final_data = resize_fn(data, new_shape, 1, **kwargs)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data.astype(dtype_data)


def compute_steps_for_sliding_window(patch_size, image_size, step_size):
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps


def load_pickle(file, mode='rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def write_pickle(obj, file, mode='wb'):
    with open(file, mode) as f:
        pickle.dump(obj, f)


save_pickle = write_pickle


def load_json(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


write_json = save_json


def crop_with_label(img, label):
    mask = np.zeros_like(label)
    mask[label != 0] = 1
    nonzeromask = create_nonzero_mask(mask)
    bbox = get_bbox_from_mask(nonzeromask)
    return crop_to_bbox(img, bbox)


def crop_zero_region(img):
    mask = np.zeros_like(img)
    mask[img != 0] = 1
    nonzeromask = create_nonzero_mask(mask)
    bbox = get_bbox_from_mask(nonzeromask)
    return crop_to_bbox(img, bbox)


def get_property_of_dicom(data_itk):
    spacing = data_itk.GetSpacing()
    origin = data_itk.GetOrigin()
    direction = data_itk.GetDirection()

    return spacing, origin, direction


def getImageConfig(imgpath, segpath, tempdir):
    image = sitk.ReadImage(imgpath)
    image_array = sitk.GetArrayFromImage(image)
    image_array = image_array - image_array.min()
    # spike_x = np.argmax(np.bincount(image_array.astype('int32').reshape(-1)))
    # image_array = image_array - spike_x
    seg = sitk.ReadImage(segpath)
    seg_array = sitk.GetArrayFromImage(seg)
    # seg_array[seg_array == 2] = 0
    # seg_array[seg_array > 2] = 1

    seg_array[seg_array > 0] = 1

    # seg_array[seg_array < 4] = 0
    # seg_array[seg_array == 4] = 1

    c = image_array.copy()
    c[seg_array == 0] = 0
    size = image_array.shape[0] * image_array.shape[1] * image_array.shape[2]
    sclrange = int(image_array.max())

    if not os.path.exists(tempdir):
        os.mkdir(tempdir)
    newct = sitk.GetImageFromArray(c)
    sitk.WriteImage(newct, os.path.join(tempdir, os.path.basename(imgpath)))

    return size, sclrange


def get_vtk_hist(filepath, sclrange):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(filepath)
    reader.Update()
    image = reader.GetOutput()
    hist = vtk.vtkImageHistogram()
    hist.SetInputData(image)
    hist.SetNumberOfBins(sclrange)
    hist.Update()
    hist = vtk_to_numpy(hist.GetHistogram())
    return hist


def save_nii(array, prop, path):
    ct_itk = sitk.GetImageFromArray(array)
    ct_itk.SetSpacing(prop[0])
    ct_itk.SetOrigin(prop[1])
    ct_itk.SetDirection(prop[2])
    sitk.WriteImage(ct_itk, path)


def read_nii_2_np(datapath):
    ct_itk = sitk.ReadImage(datapath)
    s, o, d = get_property_of_dicom(ct_itk)
    ct_array = sitk.GetArrayFromImage(ct_itk)
    prop = [s, o, d]
    return ct_array, prop


def keep_connected_regions(input_data_array, res_region_array, target_region_vals):
    label, num = meas.label(input_data_array, connectivity=3, return_num=True)
    region_size_label_val = {}
    for i in range(num):
        print(i, '/', num)
        label_val = i + 1
        region_size = (label == label_val).sum()
        region_size_label_val[region_size] = label_val

    sorted_list = sorted(region_size_label_val.items(), reverse=True)
    kept_num = 0
    for key_val in sorted_list:
        res_region_array[label == key_val[1]] = target_region_vals[kept_num]
        kept_num += 1
        if kept_num == len(target_region_vals):
            break
    return


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


def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def get_peak_mid(hist):
    hist_r = hist[::-1]
    # 找到波峰索引
    peaks_indices, _ = find_peaks(hist_r, width=10, height=0.001 * np.sum(hist))
    # 获取波峰对应的横坐标
    peak_x = len(hist_r) - peaks_indices[0]
    return peak_x


def ct_preprocess(ct_array, k):
    ct_array = ct_array.astype('float')
    ct_array = ct_array - ct_array.min()
    data_flat = ct_array.flatten()
    hist, _ = np.histogram(data_flat, bins=np.arange(np.min(data_flat), np.max(data_flat) + 1))

    hist_m = medfilt(hist, 5)
    peak_x = get_peak_mid(hist_m)
    y_data = hist_m[peak_x - 200:peak_x + 200]
    x_data = np.array(range(0, len(y_data)))
    y_data[np.isinf(y_data)] = 1
    y_data[np.isnan(y_data)] = 0

    # Fit the logistic function to the data
    optimized_parameters, _ = curve_fit(gaussian, x_data, y_data, maxfev=100000)

    A_fit, mu_fit, sigma_fit = optimized_parameters
    cut = peak_x + abs(sigma_fit) * k

    ct_array = ct_array - cut
    ct_array[ct_array < 0] = 0
    ct_array = np.clip(ct_array, ct_array.min(), np.percentile(ct_array, 99.5)).astype('float32')
    return ct_array


def get_metadata(imagepath):
    image = sitk.ReadImage(imagepath)
    metadata_dict = {'Manufacturer': image.GetMetaData('0008|0070') if image.HasMetaDataKey('0008|0070') else None,
                     'Manufacturer\'s Model Name': image.GetMetaData('0008|1090') if image.HasMetaDataKey(
                         '0008|1090') else None,
                     'Patient\'s Birth Date': image.GetMetaData('0010|0030') if image.HasMetaDataKey(
                         '0010|0030') else None,
                     'Patient\'s Sex': image.GetMetaData('0010|0040') if image.HasMetaDataKey('0010|0040') else None,

                     'KVP (Tube Voltage)': image.GetMetaData('0018|0060') if image.HasMetaDataKey(
                         '0018|0060') else None,
                     'X-Ray Tube Current (mA)': image.GetMetaData('0018|1151') if image.HasMetaDataKey(
                         '0018|1151') else None}
    return metadata_dict


def get_spacing_from_dicom(filepath):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(filepath)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    spacing = image.GetSpacing()
    return spacing


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


def remove_small_regions(input_data_array, res_region_array, min_region_size, res_label_val):
    label, num = meas.label(input_data_array, connectivity=3, return_num=True)
    for i in range(num):
        label_val = i + 1
        region_size = (label == label_val).sum()
        if region_size < min_region_size:
            continue

        res_region_array[label == label_val] = res_label_val


def bbox_expansion(bbox, pad_size, data_shape, spacing):
    expansion = [int(pad_size / spacing[i]) for i in range(3)]

    bbox = [
        [max(0, bbox[0][0] - expansion[0]), min(data_shape[0], bbox[0][1] + expansion[0])],
        [max(0, bbox[1][0] - expansion[1]), min(data_shape[1], bbox[1][1] + expansion[1])],
        [max(0, bbox[2][0] - expansion[2]), min(data_shape[2], bbox[2][1] + expansion[2])]
    ]

    return bbox


def has_aniso_spacing(ori_spacing, ori_shape, anisotropy_threshold=3):
    target = ori_spacing
    target_size = ori_shape
    worst_spacing_axis = np.argmax(target)
    other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
    other_spacings = [target[i] for i in other_axes]
    other_sizes = [target_size[i] for i in other_axes]

    has_aniso_spacing = target[worst_spacing_axis] > (anisotropy_threshold * max(other_spacings))
    has_aniso_voxels = target_size[worst_spacing_axis] * anisotropy_threshold < min(other_sizes)

    if has_aniso_voxels and has_aniso_spacing:
        return True
    else:
        return False


def fix_anisotropy(ori_spacing, ori_shape, anisotropy_threshold=3):
    target = ori_spacing
    worst_spacing_axis = np.argmax(target)
    other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
    other_spacings = [target[i] for i in other_axes]

    spacings_of_that_axis = worst_spacing_axis
    target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
    # don't let the spacing of that axis get higher than the other axes
    if target_spacing_of_that_axis < max(other_spacings):
        target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
    target[worst_spacing_axis] = target_spacing_of_that_axis

    new_shape = np.round(
        ((np.array(ori_spacing) / np.array(target)).astype(float) * np.array(ori_shape))).astype(int)
    resampled_spacing = target

    return new_shape, resampled_spacing


def resample_data2(data, new_shape):
    assert len(data.shape) == 3, "data must be (z, y, x)"
    dtype_data = data.dtype
    data = data.astype('float32')
    data = np.transpose(data, (2, 1, 0))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data[None][None]
    data = torch.tensor(data).to(device)

    out_d = new_shape[0]
    out_h = new_shape[1]
    out_w = new_shape[2]

    # 生成三维坐标网格
    new_d = torch.linspace(-1, 1, out_d).view(-1, 1, 1).repeat(1, out_h, out_w)
    new_h = torch.linspace(-1, 1, out_h).view(1, -1, 1).repeat(out_d, 1, out_w)
    new_w = torch.linspace(-1, 1, out_w).view(1, 1, -1).repeat(out_d, out_h, 1)

    # 将三个维度的坐标拼接成一个三维坐标网格
    grid = torch.cat((new_d.unsqueeze(3), new_h.unsqueeze(3), new_w.unsqueeze(3)), dim=3)
    grid = grid.unsqueeze(0).to(device)

    outp = grid_sample(data, grid=grid, mode='nearest', align_corners=True).cpu().numpy()[0][0]
    print('new shape: ', outp.shape)
    return outp.astype(dtype_data)


def resample_data_monai(data, scale_ratio):
    assert len(data.shape) == 3, "data must be (z, y, x)"
    dtype_data = data.dtype
    data = data.astype('float32')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data[None]
    data = torch.tensor(data).to(device)

    transform = monai.transforms.Compose([
        monai.transforms.Orientation(axcodes="RAS"),
        monai.transforms.Spacing(pixdim=scale_ratio, align_corners=True, mode="bilinear")
    ])
    outp = transform(data).cpu().numpy()[0]
    print('new shape: ', outp.shape)
    return outp.astype(dtype_data)


def seperate_LR(res):
    # Label结果后处理
    label_data = np.zeros(res.shape, res.dtype)
    # 处理神经管 Label = 1,保留两个最大的连通域分别设置为1和2
    label_data_1 = np.zeros(res.shape, res.dtype)
    label_data_1[res == 1] = 1
    keep_connected_regions(label_data_1, label_data, [1, 2])

    # 通过label 1和2的x轴位置区分左右神经管
    index_1 = np.where(label_data == 1)
    index_2 = np.where(label_data == 2)
    if len(index_1[2]) > 0 and len(index_2[2]) > 0:
        x_center_label_1 = index_1[2][int(len(index_1[2]) / 2)]
        x_center_label_2 = index_2[2][int(len(index_2[2]) / 2)]
        if x_center_label_1 < x_center_label_2:
            label_data[label_data == 1] = 255
            label_data[label_data == 2] = 1
            label_data[label_data == 255] = 2

    return label_data


def delete_files(folder_path, file_name):
    # 遍历文件夹内的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == file_name:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"删除文件: {file_path}")


def get_inferdata_from_mesh(mesh, decimate_target):
    if mesh.GetNumberOfCells() > decimate_target:
        ratio = decimate_target / mesh.GetNumberOfCells()
        decimate = vtk.vtkQuadricDecimation()
        decimate.SetInputData(mesh)
        decimate.SetTargetReduction(1 - ratio)
        decimate.Update()
        mesh = decimate.GetOutput()

    num_cells = mesh.GetNumberOfCells()
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
    postpro_prop = np.column_stack((cells, barycenters, normals))

    barycenters -= mean_cell_centers[0:3]

    cells[:, 0:3] -= mean_cell_centers[0:3]
    cells[:, 3:6] -= mean_cell_centers[0:3]
    cells[:, 6:9] -= mean_cell_centers[0:3]
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

    points = vtk_to_numpy(mesh.GetPoints().GetData()).copy()

    maxs = points.max(axis=0)
    mins = points.min(axis=0)
    nmeans = normals.mean(axis=0)
    nstds = normals.std(axis=0)

    means = points.mean(axis=0)
    stds = points.std(axis=0)

    nmeans1 = mesh_normals1.mean(axis=0)
    nstds1 = mesh_normals1.std(axis=0)
    nmeans2 = mesh_normals2.mean(axis=0)
    nstds2 = mesh_normals2.std(axis=0)
    nmeans3 = mesh_normals3.mean(axis=0)
    nstds3 = mesh_normals3.std(axis=0)

    # curvatures_array = (curvatures_array-curvatures_array.min())/(curvatures_array.max()-curvatures_array.min())

    for i in range(3):
        cells[:, i] = (cells[:, i] - means[i]) / stds[i]  # point 1
        cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]  # point 2
        cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]  # point 3
        normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]
        barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
        mesh_normals1[:, i] = (mesh_normals1[:, i] - nmeans1[i]) / nstds1[i]
        mesh_normals2[:, i] = (mesh_normals2[:, i] - nmeans2[i]) / nstds2[i]
        mesh_normals3[:, i] = (mesh_normals3[:, i] - nmeans3[i]) / nstds3[i]

    # X = np.column_stack((barycenters,normals,cells,mesh_normals1,mesh_normals2,mesh_normals3))
    X = np.column_stack((barycenters, normals))

    return X, mesh, postpro_prop


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_point_cloud(source_polydata, target_polydata, voxel_size):
    print("Load two point clouds and disturb initial pose.")

    points = numpy_support.vtk_to_numpy(source_polydata.GetPoints().GetData())
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(points)

    points_tooth_part = numpy_support.vtk_to_numpy(target_polydata.GetPoints().GetData())
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(points_tooth_part)

    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
    return source_pcd, target_pcd, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(3000000, 0.9999))
    return result


def registration_polydata(source_polydata, target_polydata):
    voxel_size = 1.0
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_point_cloud(
        source_polydata, target_polydata, voxel_size)
    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    # return source, target, source_down, target_down, result_ransac.transformation
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down, target_down, 0.01, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    return source, target, source_down, target_down, reg_p2p.transformation


def icp_transform(source, target, trans_matrix):
    curMatrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            curMatrix.SetElement(i, j, trans_matrix[i][j])

    curTransform = vtk.vtkTransform()
    curTransform.SetMatrix(curMatrix)

    scanTransformFilter = vtk.vtkTransformPolyDataFilter()
    scanTransformFilter.SetInputData(source)
    scanTransformFilter.SetTransform(curTransform)
    scanTransformFilter.Update()

    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(scanTransformFilter.GetOutput())
    icp.SetTarget(target)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    # icp.DebugOn()
    icp.SetMaximumNumberOfIterations(200)
    # icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    newMatrix = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Multiply4x4(icp.GetMatrix(), curMatrix, newMatrix)

    new_trans_matrix = np.copy(trans_matrix)
    for i in range(4):
        for j in range(4):
            new_trans_matrix[i][j] = newMatrix.GetElement(i, j)
    return new_trans_matrix
