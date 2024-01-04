from utils import *


def resample_data3(data,target_shape):
    ori_shape = data.shape
    dtype_data = data.dtype
    data = data.astype('float32')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data[None]
    data = torch.tensor(data).to(device)

    scale_ratio =[float(target_shape[0]/ori_shape[0]),float(target_shape[1]/ori_shape[1]),float(target_shape[2]/ori_shape[2])]
    theta = torch.tensor([[scale_ratio[0], 0, 0, 0],
                          [0, scale_ratio[1], 0, 0],
                          [0, 0, scale_ratio[2], 0]])

    grid = torch.nn.functional.affine_grid(theta,ori_shape,True)
    res = torch.nn.functional.grid_sample(data,grid,"bilinear",align_corners=True)
    outp = res.cpu().numpy()[0]
    print('new shape: ', outp.shape)
    return outp.astype(dtype_data)


label,prop = read_nii_2_np(r'C:\DL_DataBase\bone\output\label/2020f-01-pre-1.nii.gz')
label2,_ = read_nii_2_np(r'C:\DL_DataBase\bone\label/2020f-01-pre-1.nii.gz')

shape = label2.shape

res = resample_data3(label,shape)
save_nii(res,prop,r'C:\DL_DataBase\bone/a.nii.gz')