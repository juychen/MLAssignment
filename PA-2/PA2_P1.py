import itertools
import math as m
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import PA2_IMP as im

CLRS = ["c", "b", "m", "r"]


def plot_cluster(X, Y, C, title = 'title'):
    for x, y, c in zip(X, Y, C):
        plt.scatter(x, y, color=CLRS[(c.astype(int)) - 1])
    

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

    KMA = im.Kmeans(k=4)
    KMA.fit_x(X_A)
    KMA.cluster()
    KMA_predict = KMA.get_result()

    # print(KMA.z)

    #plot_cluster(X_A[:, 0], X_A[:, 1],KMA_predict ,'dataA')

    
    GMMA = im.EMGMM(k=4, itera=100)
    GMMA.fit_x(X_A)
    GMMA.cluster()
    GMA_predict = GMMA.get_result()

    print(GMMA.z)

    print(GMMA.get_result())
    
    plot_cluster(X_A[:, 0], X_A[:, 1],GMA_predict ,'dataA')

    # print(GMM.SIGMA)

    print(0)

    return


if __name__ == "__main__":
    main()
