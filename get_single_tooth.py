import os.path

import numpy as np

from utils import *

if __name__ == '__main__':

    imgpath = r'C:\DL_DataBase\CBCT_data\CTooth\NC release data\img'
    labelpath = r'C:\DL_DataBase\CBCT_data\CTooth\NC release data\label'
    outpath = r'C:\DL_DataBase\CBCT_data\raw_data\singletooth'

    datalist = os.listdir(labelpath)

    for i in datalist:
        ct_array, prop = read_nii_2_np(os.path.join(imgpath, i))
        label_array, _ = read_nii_2_np(os.path.join(labelpath, i))

        ct_array = data_preprocess(ct_array)

        print('cutting: ', i)
        labels = np.unique(label_array)[1:]
        for l in labels:
            mask = label_array == l
            if np.count_nonzero(mask.astype(int)) < 1000:
                continue
            bbox = get_bbox_from_mask(mask)
            expansion = [int(3/prop[0][2]),int(3/prop[0][1]),int(3/prop[0][0])]

            bbox = [
                [max(0, bbox[0][0] - expansion[0]), min(ct_array.shape[0], bbox[0][1] + expansion[0])],
                [max(0, bbox[1][0] - expansion[1]), min(ct_array.shape[1], bbox[1][1] + expansion[1])],
                [max(0, bbox[2][0] - expansion[2]), min(ct_array.shape[2], bbox[2][1] + expansion[2])]
            ]

            tooth_img_array = ct_array[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]: bbox[2][1]]
            tooth_label_array = label_array[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]: bbox[2][1]]
            save_label = tooth_label_array==l
            save_label = save_label.astype(int)

            out_img_path = os.path.join(outpath, 'img', '{}_{}.nii.gz'.format(i[:-7], l))
            out_label_path = os.path.join(outpath, 'label', '{}_{}.nii.gz'.format(i[:-7], l))
            save_nii(tooth_img_array, prop, out_img_path)
            save_nii(save_label, prop, out_label_path)
