import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def logistic_function(x,x0, k):
    return 1 / (1 + np.exp(-(x - x0) / k))
# x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# y_data = np.array([0.1, 0.3, 0.4, 0.7, 1.2, 2.0, 2.8, 3.5, 3.9, 4])





if __name__ == '__main__':
    datapath = r'C:\DL_DataBase\CBCT_data\raw_data\st\out\npz'
    datalist = os.listdir(datapath)

    x0=[]
    k=[]
    e=[]
    for i in datalist:
        y_data = np.load(os.path.join(datapath,i))['P']
        l = int(y_data.size)
        y_data = y_data[:l//4]
        # peak_index_data = np.load(os.path.join(datapath,i))['peak_index']
        x_data = np.array(range(0,len(y_data)))
        y_data[np.isinf(y_data)]=1
        y_data[np.isnan(y_data)]=0


        # Initial parameter estimates (L, x0, k)
        initial_parameters = [3, 1]

        # Fit the logistic function to the data
        optimized_parameters, _ = curve_fit(logistic_function, x_data, y_data, p0=initial_parameters)

        x0_opt, k_opt = optimized_parameters
        # print("x0:", x0_opt, "k:", k_opt,'peak:',peak_index_data,'e:',x0_opt - (peak_index_data/990*(1700-990)+peak_index_data))
        print("x0:", x0_opt, "k:", k_opt)
        # if x0_opt>5000:
        #     print(i)

        # x_fit = np.linspace(x_data.min(), x_data.max(), 1000)
        # y_fit = logistic_function(x_fit, x0_opt, k_opt)
        # plt.scatter(x_data, y_data, label='Data', marker='o',s=5)
        # plt.plot(x_fit, y_fit, label='Fitted logistic function', color='red')
        # plt.legend()
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.show()
        if x0_opt<5000:
            x0.append(x0_opt)
            k.append(k_opt)
            # e.append(np.abs(x0_opt - (peak_index_data/990*800+peak_index_data)))
    print(np.mean(x0))
    print(1/np.mean(k))
    # print(np.mean(e))


