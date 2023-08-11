import os
import time
from utils import *

if __name__ == '__main__':
    imgpath = r'C:\DL_DataBase\CBCT_data\CTooth\NC release data\img'
    labelpath = r'C:\DL_DataBase\CBCT_data\CTooth\NC release data\label'
    savepath = r'C:\DL_DataBase\CBCT_data\new_data\train\tooth'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    imglist = os.listdir(labelpath)
    target_spacing = [0.4, 0.4, 0.4]
    for file in imglist:
        if os.path.exists(os.path.join(savepath, 'img',file)):
            continue
        print(file)
        s = time.time()
        ct_array, prop = read_nii_2_np(os.path.join(imgpath, file))
        label_array, _ = read_nii_2_np(os.path.join(labelpath, file))
        spacing = prop[0]
        ori_data_shape = ct_array.shape

        ct_array[np.isnan(ct_array)] = 0
        ct_array = ct_array.astype('float32')
        ct_array = ct_array - ct_array.min()
        peak_x = get_peak_idx(ct_array)

        mask = label_array > 0
        roi_bbox = get_bbox_from_mask(mask)
        delta_mm = 1.5
        delta_dims = [int(delta_mm / spacing[2]), int(delta_mm / spacing[1]), int(delta_mm / spacing[0])]  # z,y,x
        for i in range(3):
            roi_bbox[i][0] = max(0, roi_bbox[i][0] - delta_dims[i])
            roi_bbox[i][1] = min(roi_bbox[i][1] + delta_dims[i], ori_data_shape[i])
        ct_array = crop_to_bbox(ct_array, roi_bbox)
        label_array = crop_to_bbox(label_array, roi_bbox)

        new_shape = np.round(
            ((np.array(spacing) / np.array(target_spacing)).astype(float) * np.array(
                ct_array.shape))).astype(int)
        ct_array = resample_data(ct_array, new_shape)
        label_array = resample_data(label_array, new_shape)

        ct_array = ct_array - peak_x
        ct_array[ct_array < 0] = 0
        ct_array = np.clip(ct_array, ct_array.min(), np.percentile(ct_array, 99.5)).astype('float32')
        prop[0] = target_spacing
        save_nii(ct_array,prop,os.path.join(savepath, 'img',file))
        save_nii(label_array,prop,os.path.join(savepath, 'label',file))
        e = time.time()
        print('time: ', e - s)
