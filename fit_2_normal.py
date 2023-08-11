import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, medfilt

from utils import *


# def clipping_filter(signal, max_deviation):
#     filtered_signal = [0]  # 初始化过滤后的信号列表，将第一个采样值加入其中
#     for i in range(1, len(signal)):
#         deviation = signal[i] - signal[i - 1]  # 计算相邻采样值之间的差值
#         if deviation > max_deviation:  # 如果差值超过最大允许误差
#             filtered_signal.append(signal[i - 1])  # 将前一个采样值加入过滤后的信号列表
#         elif deviation < -1 * max_deviation:
#             filtered_signal.append(signal[i])
#         else:
#             filtered_signal.append(signal[i])  # 将当前采样值加入过滤后的信号列表
#
#     return np.array(filtered_signal)

def clipping_filter(signal, max_deviation):
    for i in range(0, len(signal) - 1):
        deviation = signal[i + 1] - signal[i]  # 计算相邻采样值之间的差值
        if deviation > max_deviation:  # 如果差值超过最大允许误差
            signal[i + 1] = signal[i]
            # filtered_signal.append(signal[i - 1])  # 将前一个采样值加入过滤后的信号列表
        elif deviation < -1 * max_deviation:
            signal[i] = signal[i + 1]


def get_peak_mid(hist):
    hist_r = hist[::-1]
    # 找到波峰索引
    peaks_indices, _ = find_peaks(hist_r, width=10, height=0.001*np.sum(hist))
    # 获取波峰对应的横坐标
    peak_x = len(hist_r) - peaks_indices[0]
    return peak_x
def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

imgpath = r'C:\DL_DataBase\CBCT_data\raw_data\png\npz'
savepath = r'C:\DL_DataBase\CBCT_data\raw_data\png\fit'
if not os.path.exists(savepath):
    os.mkdir(savepath)
imglist = os.listdir(imgpath)
for i in imglist:
    img_savepath = os.path.join(savepath, '{}.png'.format(i[:-4]))
    # if os.path.exists(img_savepath):
    #     continue
    print(i)
    array = np.load(os.path.join(imgpath, i))
    hist = array['hist']
    # bin_edges = array['bin_edges']
    hist_m = medfilt(hist, 5)
    peak_x = get_peak_mid(hist_m)

    y_data = hist_m[peak_x - 200:peak_x + 200]
    x_data = np.array(range(0, len(y_data)))
    y_data[np.isinf(y_data)] = 1
    y_data[np.isnan(y_data)] = 0

    # Fit the logistic function to the data
    optimized_parameters, _ = curve_fit(gaussian, x_data, y_data, maxfev=100000)

    A_fit, mu_fit, sigma_fit = optimized_parameters

    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax = fig.add_subplot()
    # 绘制数组数据
    ax.plot(hist, label='ori', color='blue')
    ax.plot(hist_m, label='after_median_filter', color='red',linestyle='--')
    ax.set_ylim(0, 500000)
    # 添加标题和标签
    ax.set_title('CBCT Hist')
    ax.set_xlabel('Gray Scale')
    ax.set_ylabel('Frequence')
    x_data = np.array(range(0, len(hist_m)))
    plt.plot(x_data, gaussian(x_data, A_fit, peak_x, sigma_fit), 'g--', label='Fitted Gaussian')
    ax.vlines([peak_x], 0, hist_m.max(), linestyles='dashed', colors='black')
    # plt.show()
    plt.savefig(img_savepath)

# 显示图形
# plt.show()
