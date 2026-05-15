import os
import numpy as np
import SimpleITK as sitk
from PIL import Image



def convert_nifti_to_png(input_dir, output_dir):
    """
    将单层NIfTI文件转换为PNG图像
    
    Args:
        input_dir: 包含.nii.gz文件的输入目录
        output_dir: 输出PNG文件的目录
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Supported nifti extensions
    nifti_extensions = ('.nii.gz', '.nii')

    # Process each nifti file
    converted_count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(nifti_extensions):
            try:
                # Read nifti file
                nii_path = os.path.join(input_dir, filename)
                sitk_img = sitk.ReadImage(nii_path)
                
                # Convert to numpy array
                img_array = sitk.GetArrayFromImage(sitk_img)
                
                # Handle different array shapes
                if img_array.ndim == 3:
                    # If 3D array, take the first slice or squeeze if single slice
                    if img_array.shape[0] == 1:
                        img_array = img_array[0]
                    else:
                        # Take middle slice if multiple slices
                        middle_idx = img_array.shape[0] // 2
                        img_array = img_array[middle_idx]
                        print(f"Warning: {filename} has {img_array.shape[0]} slices, using middle slice")
                elif img_array.ndim == 2:
                    # Already 2D, use as is
                    pass
                else:
                    print(f"Warning: Unsupported array shape {img_array.shape} for {filename}, skipping")
                    continue
                
                # Normalize to 0-255 range for PNG
                if img_array.max() != img_array.min():
                    img_normalized = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                else:
                    img_normalized = np.zeros_like(img_array, dtype=np.uint8)
                
                # Create PIL Image and save as PNG
                img_pil = Image.fromarray(img_normalized)
                output_filename = os.path.splitext(filename)[0] + '.png'
                if filename.endswith('.nii.gz'):
                    output_filename = filename[:-7] + '.png'
                output_path = os.path.join(output_dir, output_filename)
                img_pil.save(output_path)
                
                converted_count += 1
                print(f"Converted: {filename} -> {output_filename}")
                
            except Exception as e:
                print(f"Error converting {filename}: {str(e)}")

    print(f'\nSuccessfully converted {converted_count} NIfTI files to PNG format in {output_dir}')


def convert_images_to_nifti(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Supported image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    # Process each image file
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            # Read image
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale

            # Convert to numpy array and add channel dimension [1, height, width]
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # img_array[img_array > 0] = 1

            # Create SimpleITK image
            sitk_img = sitk.GetImageFromArray(img_array)

            # Save as NIfTI compressed format
            output_filename = os.path.splitext(filename)[0] + '.nii.gz'
            output_path = os.path.join(output_dir, output_filename)
            sitk.WriteImage(sitk_img, output_path)

    print(f'Converted {len([f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)])} images to NIfTI format in {output_dir}')

# convert_images_to_nifti(r'C:\DL_DataBase\Img_data\001\image',r'C:\DL_DataBase\Img_data\001\image_nii')
convert_nifti_to_png(r'C:\DL_DataBase\Img_data\001\tooth_nii',r'C:\DL_DataBase\Img_data\001\tooth')