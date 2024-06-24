import os.path

from utils import *

datapath = r'C:\DL_DataBase\CBCT_data\skull\ddd'
roipath = r'C:\DL_DataBase\CBCT_data\skull\new_bone'
labelpath = r'C:\DL_DataBase\CBCT_data\skull\new_zygoma'
outpath = r'C:\DL_DataBase\CBCT_data\skull'

use_roi = False
target_spacing = [0.25, 0.25, 0.25]
pad_size = 3

for i in os.listdir(datapath):
    if os.path.exists(os.path.join(outpath, 'img', i)):
        continue
    print(i)
    ct_array, prop = read_nii_2_np(os.path.join(datapath, i))
    label, _ = read_nii_2_np(os.path.join(labelpath, i))

    ct_array[np.isnan(ct_array)] = 0
    ct_array = ct_array.astype('float32')
    ct_array = ct_array - ct_array.min()
    label = label.astype('uint8')

    if use_roi:
        roi, _ = read_nii_2_np(os.path.join(roipath, i))
        roi[roi != 1] = 0
        bbox = get_bbox_from_mask(roi)
        expansion = [int(pad_size / prop[0][2]), int(pad_size / prop[0][1]), int(pad_size / prop[0][0])]
        bbox = [
            [max(0, bbox[0][0] - expansion[0]), min(label.shape[0], bbox[0][1] + expansion[0])],
            [max(0, bbox[1][0] - expansion[1]), min(label.shape[1], bbox[1][1] + expansion[1])],
            [max(0, bbox[2][0] - expansion[2]), min(label.shape[2], bbox[2][1] + expansion[2])]
        ]

        ct_array = crop_to_bbox(ct_array, bbox)
        label = crop_to_bbox(label, bbox)

    new_shape = np.round(
        ((np.array(prop[0]) / np.array(target_spacing)).astype(float) * np.array(
            ct_array.shape))).astype(int)
    ct_array = resample_data(ct_array, new_shape)
    label = resample_data(label, new_shape)

    peak_x = get_peak_idx(ct_array)
    ct_array = ct_array - peak_x
    ct_array[ct_array < 0] = 0

    ct_array = np.clip(ct_array, ct_array.min(), np.percentile(ct_array, 99.5)).astype('float32')
    prop[0] = target_spacing

    save_nii(ct_array, prop, os.path.join(outpath, 'img', i))
    save_nii(label, prop, os.path.join(outpath, 'label', i))
