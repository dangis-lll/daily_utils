import numpy as np
from utils import *
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure, sobel
from skimage import morphology
from skimage.morphology import square, cube

def extract_surface_sobel(binary_image):
    # 计算三个方向上的梯度
    dx = sobel(binary_image, axis=0)
    dy = sobel(binary_image, axis=1)
    dz = sobel(binary_image, axis=2)

    # 计算梯度的幅度
    magnitude = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # 将梯度幅度大于零的体素设置为1（表示物体的表面）
    surface = (magnitude > 0).astype(np.uint8)

    return surface


def extract_surface_morph(binary_image):
    # 创建一个用于膨胀和腐蚀操作的结构元素
    struct_element = generate_binary_structure(3, 1)

    # 对二值图像进行膨胀操作
    dilated_image = binary_dilation(binary_image, structure=struct_element)

    # 对二值图像进行腐蚀操作
    eroded_image = binary_erosion(binary_image, structure=struct_element)

    # 计算物体表面：膨胀图像与腐蚀图像的差值
    surface = dilated_image - eroded_image

    return surface

path = r'C:\DL_DataBase\CBCT_data\alltooth\fine_data\label/01-001-ZYHA.nii.gz'
ct_array,prop = read_nii_2_np(path)

ct_array = ct_array.astype('uint8')

labels = np.unique(ct_array)[1:]

surface = np.zeros_like(ct_array)

for i in labels:
    mask = np.zeros_like(ct_array)
    mask[ct_array==i]=1

    struct_element = generate_binary_structure(3, 1)
    eroded_image = binary_erosion(mask, structure=struct_element).astype('uint8')
    eroded_image = mask-eroded_image
    surface[eroded_image==1]=1

    # mask_ = morphology.erosion(mask, cube(2))
    # mask[mask_==1]=0
    # surface[mask==1]=1


save_nii(surface,prop,'a.nii.gz')
