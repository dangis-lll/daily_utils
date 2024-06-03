from utils import *

datapath = r'C:\DL_DataBase\CBCT_data\skull'
outpath = r'C:\DL_DataBase\CBCT_data\skull\sparse_loc'


for i in os.listdir(os.path.join(datapath,'label')):
    ct,prop = read_nii_2_np(os.path.join(datapath,'img',i))
    label,_ = read_nii_2_np(os.path.join(datapath,'label',i))
    label[label==4]=3
    label[label>3]=4
    ct = ct.astype(np.float32)
    ct = ct-ct.min()
    peak = get_peak_idx(ct)
    target_spacing = [0.6, 0.6, 0.6]
    new_shape = np.round(
        ((np.array(prop[0]) / np.array(target_spacing)).astype(float) * np.array(
            ct.shape))).astype(int)
    ct = resample_data(ct, new_shape)
    label = resample_data(label, new_shape)

    prop[0] = target_spacing
    ct = ct - peak
    ct[ct<0]=0

    print(i)
    save_nii(ct,prop,os.path.join(outpath,'img',i))
    save_nii(label,prop,os.path.join(outpath,'label',i))