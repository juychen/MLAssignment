import itertools
import math as m
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import PA2_IMP as im
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer



CLRS = ["c", "b", "m", "r"]
DATA = ['A', 'B', 'C']


def plot_cluster(X, Y, C, title='title'):
    for x, y, c in zip(X, Y, C):
        plt.scatter(x, y, color=CLRS[(c.astype(int)) - 1])

    plt.title(title)
    plt.axis([-15, 15, -15, 15])
    plt.show()
    plt.close()
    return

def plot_MS(X, Y, C, title='title'):
    for x, y, c in zip(X, Y, C):
        plt.scatter(x, y, c=[[c[0],c[1],0]])

    plt.title(title)
    plt.axis([-15, 15, -15, 15])
    plt.show()
    plt.close()
    return

def main():

    X_A = im.load_file(filename=os.path.join(
        'PA-2', 'data', 'cluster_data_dataA_X.txt')).T
    Y_A = im.load_file(filename=os.path.join(
        'PA-2', 'data', 'cluster_data_dataA_Y.txt')).T
    X_B = im.load_file(filename=os.path.join(
        'PA-2', 'data', 'cluster_data_dataB_X.txt')).T
    Y_B = im.load_file(filename=os.path.join(
        'PA-2', 'data', 'cluster_data_dataB_Y.txt')).T
    X_C = im.load_file(filename=os.path.join(
        'PA-2', 'data', 'cluster_data_dataC_X.txt')).T
    Y_C = im.load_file(filename=os.path.join(
        'PA-2', 'data', 'cluster_data_dataC_Y.txt')).T

    exp_dict = {}

    # for data in DATA:
    #     exp_dict[data]['X'] = im.load_file(filename=os.path.join(
    #         'PA-2', 'data', 'cluster_data_data' + data + '_X.txt')).T
    #     exp_dict[data]['Y'] = im.load_file(filename=os.path.join(
    #         'PA-2', 'data', 'cluster_data_data' + data + '_Y.txt')).T

    #     KM = im.Kmeans(k=4)
    #     KM.fit_x(exp_dict[data]['X'])
    #     KM.cluster()
    #     exp_dict[data]['KM'] = KM

    #     GMM = im.EMGMM(k=4)
    #     GMM.fit_x(exp_dict[data]['X'])
    #     GMM.cluster()
    #     exp_dict[data]['GMM'] = GMM

    KMA = im.Kmeans(k=4)
    KMA.fit_x(X_A)
    KMA.cluster()
    KMA_predict = KMA.get_result()

    # print(KMA.z)

    # plot_cluster(X_A[:, 0], X_A[:, 1],KMA_predict ,'dataA')

    GMMA = im.EMGMM(k=4, itera=100)
    GMMA.fit_x(X_A)
    GMMA.cluster()
    GMA_predict = GMMA.get_result()

    #print(GMMA.z)

    print(GMMA.get_result())

    #plot_cluster(X_A[:, 0], X_A[:, 1], GMA_predict, 'dataA')
    # print (Y_A)

    # print(GMM.SIGMA)

    GMSA = im.GaussianMeanShift(bandwidth=5)
    GMSA.fit_x(X_A)
    GMSA.cluster()
    print(GMSA.x_)

    #plot_MS(X_A[:, 0], X_A[:, 1], rgclrs, 'dataA')


    print(0)

    return


if __name__ == "__main__":
    main()
