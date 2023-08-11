import math

import SimpleITK as sitk
import os

import numpy as np
import skimage
from matplotlib import pyplot as plt

if __name__ == '__main__':
    datapath = r'plt_output\resampled\hist_npz'
    # datapath = r'C:\DL_DataBase\CBCT_data\alltooth\output\hist_npz'
    fig = plt.figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot()
    ax.set_ylim(0, 0.2)
    datalist = os.listdir(datapath)
    colors = plt.get_cmap('RdBu', len(datalist))([i for i in range(len(datalist))])
    c = 0
    lens = 0
    for i in os.listdir(datapath):
        data = os.path.join(datapath, i)

        P = np.load(data)['P']
        peak_index = np.load(data)['peak_index']
        lens = len(P)
        ax.plot(P, color=colors[c])
        Y = [0] * lens
        for i in range(lens):
            Y[i] = 1 / (1 + math.exp(-0.007 * (i - (peak_index/980*1600))))
        ax.plot(Y, color=colors[c], linewidth=2.0)
        ax.vlines(round((peak_index/980*1600) - (math.log(pow(0.05, -1), math.e) / 0.004)), 0, 1, linestyles='dashed', color=colors[c])
        # ax.vlines(peak_index, 0, 1, linestyles='dashed', colors='yellow')
        # ax.set_ylim(0, 0.003)
        c = c + 1

    # Y = [0]*lens
    # for i in range(lens):
    #     Y[i] = 1/(1+math.exp(-0.006*(i-1400)))
    # ax.plot(Y,color = 'red',linewidth =5.0)
    # ax.hlines(0.01,0,1000,  linestyles='dashed', colors='black')
    ax.set_xlim(0, 1500)
    plt.grid()
    # plt.savefig('after_preprocess_hist.png')
    plt.show()