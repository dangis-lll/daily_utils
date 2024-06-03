# from scipy import ndimage
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# from scipy.signal import convolve
#
#
# def DiscreteRadonTransform(image, steps):
#     channels = len(image[0])
#     res = np.zeros((channels, channels), dtype='float64')
#     for s in range(steps):
#         rotation = ndimage.rotate(image, -s * 180 / steps, reshape=False).astype('float64')
#         # print(sum(rotation).shape)
#         res[:, s] = np.log(np.abs(np.fft.fft(sum(rotation))))
#     return res
#
#
# def RLFilter(N, d):
#     filterRL = np.zeros((N,))
#     for i in range(N):
#         filterRL[i] = - 1.0 / np.power((i - N / 2) * np.pi * d, 2.0)
#         if np.mod(i - N / 2, 2) == 0:
#             filterRL[i] = 0
#     filterRL[int(N / 2)] = 1 / (4 * np.power(d, 2.0))
#     return filterRL
#
#
# def SLFilter(N, d):
#     filterSL = np.zeros((N,))
#     for i in range(N):
#         # filterSL[i] = - 2 / (np.power(np.pi, 2.0) * np.power(d, 2.0) * (np.power((4 * (i - N / 2)), 2.0) - 1))
#         filterSL[i] = - 2 / (np.pi ** 2.0 * d ** 2.0 * (4 * (i - N / 2) ** 2.0 - 1))
#     return filterSL
#
#
# def IRandonTransform(image, steps):
#     # 定义用于存储重建后的图像的数组
#     channels = len(image[0])
#     origin = np.zeros((steps, channels, channels))
#     # filter = RLFilter(channels, 1)
#     filter = SLFilter(channels, 1)
#     for i in range(steps):
#         projectionValue = image[:, i]
#         projectionValueFiltered = convolve(filter, projectionValue, "same")
#         projectionValueExpandDim = np.expand_dims(projectionValueFiltered, axis=0)
#         projectionValueRepat = projectionValueExpandDim.repeat(channels, axis=0)
#         origin[i] = ndimage.rotate(projectionValueRepat, i * 180 / steps, reshape=False).astype(np.float64)
#     iradon = np.sum(origin, axis=0)
#     return iradon
#
#
# def transform_array(original_array):
#     # 获取原数组的大小
#     rows, cols = original_array.shape
#
#     # 创建新数组
#     transformed_array = np.zeros((rows, cols))
#
#     # 计算Y轴最大值
#     Y = rows - 1
#
#     # 遍历新数组中的每个位置
#     for u in range(rows):
#         for v in range(cols):
#             # 计算原数组中的坐标
#             x = np.arctan2(v - cols / 2, u - rows / 2)
#             y = np.sqrt((u - rows / 2) ** 2 + (v - cols / 2) ** 2) * Y / (rows / 2)
#
#             # 在原数组中获取相应位置的值，并填充到新数组中
#             if 0 <= x < cols and 0 <= y < rows:
#                 transformed_array[u, v] = original_array[int(np.round(y)), int(np.round(x))]
#
#     return transformed_array
#
# # 读取原始图片
# image = cv2.imread(r'C:\daily_utils\img/Untitled.png', cv2.IMREAD_GRAYSCALE)
# radon = DiscreteRadonTransform(image, len(image[0]))
# radon_ff  = transform_array(radon)
# image_ff = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
#
# # fbp = IRandonTransform(radon, len(radon[0]))
#
# # 绘制原始图像和对应的sinogram图
# plt.subplot(1, 3, 1)
# plt.imshow(image, cmap='gray')
# plt.subplot(1, 3, 2)
# plt.imshow(radon_ff, cmap='gray')
# plt.subplot(1, 3, 3)
# plt.imshow(np.log(np.abs(image_ff)), cmap='gray')
# plt.show()