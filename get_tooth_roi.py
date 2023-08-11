import os.path

from utils import *

if __name__ == '__main__':
    target_spacing = [.4, .4, .4]
    imgpath = r'C:\DL_DataBase\CBCT_data\raw_data\img'
    toothpath = r'C:\DL_DataBase\CBCT_data\raw_data\bone_label'
    labelpath = r'C:\DL_DataBase\CBCT_data\alltooth\dd'
    outpath = r'C:\DL_DataBase\CBCT_data\alltooth'
    datalist = os.listdir(labelpath)
    for name in datalist:
        print('cutting: ', name)
        # if not os.path.exists(os.path.join(imgpath, name)):
        #     print('pass: ', name)
        #     continue
        ct_array, prop = read_nii_2_np(os.path.join(imgpath, name))
        label_array, _ = read_nii_2_np(os.path.join(toothpath, name))

        ct_array = ct_array.astype('float32')
        ct_array = ct_array - ct_array.min()

        mask = label_array > 3
        bbox = get_bbox_from_mask(mask)
        ct_array = crop_to_bbox(ct_array, bbox)


        # label_array = resample_data_or_seg_slow(label_array, new_shape, 1, 1)

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

        new_shape = np.round(
            ((np.array(prop[0]) / np.array(target_spacing)).astype(float) * np.array(
                ct_array.shape))).astype(int)
        ct_array = resample_data(ct_array, new_shape)

        save_nii(ct_array, prop, os.path.join(outpath, 'img', name))
        np.savez(os.path.join(outpath, 'config', '{}.npz'.format(name[:-7])), peak=peak_x, sigma=sigma_fit)
