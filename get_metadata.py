import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks, argrelextrema

from utils import *

bpath = r'C:\DL_DataBase\CBCT_data\femur\img'
brandname = os.listdir(bpath)
META = {}
for j in brandname:
    datapath = os.path.join(bpath,j)
    datalist = os.listdir(datapath)
    for i in datalist:
        if i.endswith('.stl'):
            continue
        s = time.time()
        print('get property: ', i)
        imagepath = os.path.join(datapath,i,os.listdir(os.path.join(datapath,i))[0])
        filepath = os.path.join(datapath,i)
        metadata_dict = get_metadata(imagepath)
        spacing = get_spacing_from_dicom(filepath)
        metadata_dict['Spacing']=spacing
        META[str(i)] = metadata_dict
        # 获取元数据
        e = time.time()
        print('time: ', e - s)
df = pd.DataFrame(META).T

# 重置index，将index作为一列
df.reset_index(inplace=True)

# 将index列重命名为'name'
df.rename(columns = {'index':'filename'}, inplace = True)

# 保存为CSV文件，包括index
df.to_csv(r'C:\DL_DataBase\CBCT_data\femur/output.csv', index=False)
