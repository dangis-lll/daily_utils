import math
from collections import Counter

import SimpleITK as sitk
import os

import numpy as np
import skimage
from matplotlib import pyplot as plt
import cv2
import vtk
from scipy.signal import find_peaks
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.util.numpy_support import numpy_to_vtk
from scipy.optimize import curve_fit


def getImageConfig(imgpath, segpath, tempdir):
    image = sitk.ReadImage(imgpath)
    image_array = sitk.GetArrayFromImage(image)
    image_array = image_array - image_array.min()
    # spike_x = np.argmax(np.bincount(image_array.astype('int32').reshape(-1)))
    # image_array = image_array - spike_x
    seg = sitk.ReadImage(segpath)
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array[seg_array == 2] = 0
    seg_array[seg_array > 2] = 1

    # seg_array[seg_array < 4] = 0
    # seg_array[seg_array > 3] = 1

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


def logistic_function(x, x0, k):
    return 1 / (1 + np.exp(-(x - x0) / k))



# from utils import *
# path = r'C:\DL_DataBase\CBCT_data\CTooth\NC release data\img'
# imglist = os.listdir(path)
# for i in imglist:
    # imgpath = os.path.join(path, i)
    # npz_savepath = os.path.join(r'C:\DL_DataBase\CBCT_data\raw_data\png\npz', '{}.npz'.format(i[:-7]))
    # if os.path.exists(npz_savepath):
        # continue
    # print(i)
    # ct_array, _ = read_nii_2_np(imgpath)
    # ct_array = ct_array-ct_array.min()
    # data_flat = ct_array.flatten()
    # hist, bin_edges = np.histogram(data_flat, bins=np.arange(np.min(data_flat), np.max(data_flat) + 1))
    # np.savez(npz_savepath,hist=hist,bin_edges=bin_edges)




if __name__ == '__main__':

    branddir = 'statisc/resampled'
    outputdir = 'plt_output/resampled'
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    brandlist = os.listdir(branddir)
    if len(brandlist):
        for brand in brandlist:
            # if brand !='sanxing':
            #     continue
            filepath = os.path.join(branddir, brand, 'img')
            count = 1
            datalist = os.listdir(filepath)
            if len(datalist):
                for data in datalist:
                    filename = data
                    print('processing: ', filename)
                    datapath = os.path.join(filepath, filename)
                    labelpath = os.path.join(filepath.replace('img', 'label'), filename)

                    ctarray = sitk.GetArrayFromImage(sitk.ReadImage(datapath))

                    size, sclrange = getImageConfig(datapath, labelpath, outputdir)

                    ct_hist = get_vtk_hist(datapath, sclrange).astype('float32') / size
                    bone_hist = get_vtk_hist(os.path.join(outputdir, filename), sclrange).astype('float32') / size
                    P = bone_hist / ct_hist
                    P[0] = 0
                    bone_hist[0] = 0

                    os.remove(os.path.join(outputdir, filename))

                    fig = plt.figure(figsize=(10, 6), dpi=300)
                    ax = fig.add_subplot()
                    ax.plot(ct_hist, label='ct', color='blue')
                    ax.set_ylim(0, 0.005)
                    ax.plot(bone_hist, color='yellow', label='seg')
                    ax2 = ax.twinx()
                    ax2.plot(P, color='red', label='p')
                    ax2.set_ylim(0, 1.0)

                    peaks_indices, _ = find_peaks(ct_hist)
                    # 设置一个阈值，用于排除0位置附近的高点
                    threshold = 300
                    # 仅保留大于阈值的波峰
                    peaks_indices = peaks_indices[peaks_indices > threshold]
                    # 从剩余的波峰中找到最大的一个
                    peak_index = peaks_indices[np.argmax(ct_hist[peaks_indices])]

                    x = round((peak_index / 990 * 1600) - (math.log(pow(0.05, -1), math.e) / 0.004))

                    ax2.vlines([peak_index, peak_index / 980 * 720 + peak_index], 0, 1, linestyles='dashed', colors='black')
                    plt.grid()
                    plt.savefig(os.path.join(outputdir, 'hist_png', '{}_0{}_hist.png'.format(brand, count)))
                    np.savez(os.path.join(outputdir, 'hist_npz', '{}_0{}_hist.npz'.format(brand, count)),
                             ct_hist=ct_hist,
                             bone_hist=bone_hist,
                             P=P,
                             peak_index=peak_index,
                             x=x)
                    # plt.show()
                    count += 1
