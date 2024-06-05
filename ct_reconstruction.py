import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2


def central_slice_theorem(image):
    # 进行二维傅立叶变换
    fft_image = fft2(image)

    # 提取中心切片
    center_x, center_y = np.array(fft_image.shape) // 2
    center_slice = fft_image[center_x, center_y]

    # 逆傅立叶变换
    reconstructed_image = ifft2(center_slice)

    return np.abs(reconstructed_image)


# 生成示例图像
size = 128
x = np.linspace(-5, 5, size)
y = np.linspace(-5, 5, size)
X, Y = np.meshgrid(x, y)
image = np.exp(-(X ** 2 + Y ** 2))

# 使用中心切片定理重建图像
reconstructed_image = central_slice_theorem(image)

# 绘制原始图像和重建图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')

plt.show()
