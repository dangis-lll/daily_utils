from utils import *
path = r'C:\DL_DataBase\CBCT_data\CTooth\NC release data\img'
imglist = os.listdir(path)
for i in imglist:
    imgpath = os.path.join(path, i)
    npz_savepath = os.path.join(r'C:\DL_DataBase\CBCT_data\raw_data\png\npz', '{}.npz'.format(i[:-7]))
    if os.path.exists(npz_savepath):
        continue
    print(i)
    ct_array, _ = read_nii_2_np(imgpath)
    ct_array = ct_array-ct_array.min()
    data_flat = ct_array.flatten()
    hist, bin_edges = np.histogram(data_flat, bins=np.arange(np.min(data_flat), np.max(data_flat) + 1))
    np.savez(npz_savepath,hist=hist,bin_edges=bin_edges)
