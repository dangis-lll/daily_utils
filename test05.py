import os

def delete_files(folder_path, file_name):
    # 遍历文件夹内的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == file_name:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"删除文件: {file_path}")

# 指定要删除的文件名和文件夹路径
file_to_delete = 'label.nii.gz'
folder_to_search = r'C:\DL_DataBase\CBCT_data\femur\Totalsegmentator_dataset_v201'  # 请替换为实际的文件夹路径

# 删除文件
delete_files(folder_to_search, file_to_delete)
