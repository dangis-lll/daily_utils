import json
import pickle
from collections import OrderedDict

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
from vtkmodules.util.numpy_support import vtk_to_numpy
from skimage import measure as meas


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
                ct_array.shaoe))).astype(int)
        ct_array = resample_data(ct_array, new_shape)
    ct_array = ct_array - peak_x
    ct_array[ct_array < 0] = 0
    ct_array = np.clip(ct_array, ct_array.min(), np.percentile(ct_array, 99.5)).astype('float32')
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
