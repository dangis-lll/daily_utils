import os.path

from utils import *

if __name__ == '__main__':
    imgpath = r'C:\DL_DataBase\CBCT_data\raw_data\ianseg\img'
    # toothpath = r'C:\DL_DataBase\CBCT_data\raw_data\bone_label'
    # ianpath = r'C:\DL_DataBase\CBCT_data\raw_data\ianseg\IAN_label'
    outpath = r'C:\DL_DataBase\CBCT_data\raw_data\ianseg\train'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
        os.mkdir(os.path.join(outpath, 'img'))
        os.mkdir(os.path.join(outpath, 'label'))
    datalist = os.listdir(imgpath)
    for name in datalist:
        print('cutting: ', name)
        if os.path.exists(os.path.join(outpath, 'img', name)):
            print('pass: ', name)
            continue
        ct_array, prop = read_nii_2_np(os.path.join(imgpath, name))
        # label_array, _ = read_nii_2_np(os.path.join(toothpath, name))
        # ian_array, _ = read_nii_2_np(os.path.join(ianpath, name))

        ct_array = ct_array.astype('float32')
        ct_array = ct_array - ct_array.min()
        data_flat = ct_array.flatten()
        hist, _ = np.histogram(data_flat, bins=np.arange(np.min(data_flat), np.max(data_flat) + 1))
        hist_m = medfilt(hist, 5)
        peak_x = get_peak_mid(hist_m)
        ct_array = ct_array - peak_x
        ct_array[ct_array < 0] = 0
        ct_array = np.clip(ct_array, ct_array.min(), np.percentile(ct_array, 99.9)).astype('float32')

        # mask = label_array == 3
        # bbox = get_bbox_from_mask(mask)
        #
        # ct_array = crop_to_bbox(ct_array, bbox)
        # ian_array = crop_to_bbox(ian_array, bbox)

        save_nii(ct_array, prop, os.path.join(outpath,'img', name))
        # save_nii(ian_array, prop, os.path.join(outpath,'label', name))

        # z = ct_array.shape[2]
        # ct_l = ct_array[:, :, :z // 2]
        # ct_r = ct_array[:, :, z // 2:]
        # # label_l = ian_array[:, :, :z // 2]
        # # label_r = ian_array[:, :, z // 2:]
        # filename = name[:-7]
        # save_nii(ct_l, prop, os.path.join(outpath, '{}_l.nii.gz'.format(filename)))
        # save_nii(ct_r, prop, os.path.join(outpath, '{}_r.nii.gz'.format(filename)))
        # save_nii(label_l, prop, os.path.join(outpath, 'label/{}_l.nii.gz'.format(filename)))
        # save_nii(label_r, prop, os.path.join(outpath, 'label/{}_r.nii.gz'.format(filename)))
