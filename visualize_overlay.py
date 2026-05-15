import SimpleITK as sitk
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

image_dir = r"c:\DL_DataBase\Img_data\tooth\image_nii"
predict_dirs = [
    r"c:\DL_DataBase\Img_data\tooth\predict_man",
    r"c:\DL_DataBase\Img_data\tooth\predict_max",
    r"c:\DL_DataBase\Img_data\tooth\predict_sinus",
    r"c:\DL_DataBase\Img_data\tooth\tooth_nii"
]

image_files = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
selected_file = random.choice(image_files)
print(f"随机选择的病例: {selected_file}")

image_path = os.path.join(image_dir, selected_file)
image_sitk = sitk.ReadImage(image_path)
image_array = sitk.GetArrayFromImage(image_sitk)

print(f"图像形状: {image_array.shape}")
print(f"图像数值范围: [{image_array.min():.2f}, {image_array.max():.2f}]")

image_2d = np.max(image_array, axis=0)
print(f"最大密度投影后形状: {image_2d.shape}")

image_normalized = (image_2d - image_2d.min()) / (image_2d.max() - image_2d.min() + 1e-8)

colors = [
    [1.0, 0.0, 0.0, 0.5],
    [0.0, 1.0, 0.0, 0.5],
    [0.0, 0.0, 1.0, 0.5],
    [1.0, 1.0, 0.0, 0.5],
]

predict_names = ['predict_man', 'predict_max', 'predict_sinus', 'predict_tooth']




for _ in range(5):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image_normalized, cmap='gray')

    for idx, predict_dir in enumerate(predict_dirs):
        predict_path = os.path.join(predict_dir, selected_file)
        
        if os.path.exists(predict_path):
            predict_sitk = sitk.ReadImage(predict_path)
            predict_array = sitk.GetArrayFromImage(predict_sitk)
            
            predict_2d = np.max(predict_array, axis=0)
            
            mask = predict_2d > 0
            
            if mask.sum() > 0:
                colored_mask = np.zeros((*predict_2d.shape, 4))
                colored_mask[mask] = colors[idx]
                ax.imshow(colored_mask)
                print(f"{predict_names[idx]}: 找到分割区域，像素数: {mask.sum()}")
            else:
                print(f"{predict_names[idx]}: 未找到分割区域")
        else:
            print(f"{predict_names[idx]}: 文件不存在")

    ax.set_title(f'Case: {selected_file}', fontsize=14)
    ax.axis('off')

    legend_elements = []
    for idx, name in enumerate(predict_names):
        if os.path.exists(os.path.join(predict_dirs[idx], selected_file)):
            legend_elements.append(plt.Rectangle((0,0), 1, 1, 
                                                facecolor=colors[idx], 
                                                label=name))

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    output_path = os.path.join(r"c:\DL_DataBase\Img_data\tooth", 
                            f"overlay_{selected_file.replace('.nii.gz', '.png')}")
    plt.savefig(output_path, dpi=450, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"\n叠加图像已保存到: {output_path}")
