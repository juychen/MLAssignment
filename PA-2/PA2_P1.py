import itertools
import math as m
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import PA2_IMP as im
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer



CLRS = ["g", "b", "m", "r"]
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

    exp_dict = {}

    for data in DATA:
        X = im.load_file(filename=os.path.join(
            'PA-2', 'data', 'cluster_data_data' + data + '_X.txt')).T
        exp_dict[(data,'X')] = X

        Y = im.load_file(filename=os.path.join(
            'PA-2', 'data', 'cluster_data_data' + data + '_Y.txt')).T
        exp_dict[(data,'Y')] = Y

        KM = im.Kmeans(k=4)
        KM.fit_x(X)
        KM.cluster()
        exp_dict[(data,'KM')] = KM

        plot_cluster(X[:, 0], X[:, 1],KM.get_result() ,'data'+data+'_KM')

        GMM = im.EMGMM(k=4)
        GMM.fit_x(X)
        GMM.cluster()
        exp_dict[(data,'GMM')] = GMM
        plot_cluster(X[:, 0], X[:, 1],GMM.get_result() ,'data'+data+'_GMM')


        GMS = im.GaussianMeanShift(bandwidth=1)
        GMS.fit_x(X)
        GMS.cluster()
        exp_dict[(data,'GMS')] = GMS
        scl = MinMaxScaler()
        clrs = scl.fit_transform(GMS.x_)
        plot_MS(X[:, 0], X[:, 1], clrs, 'data'+data+'_GMS')

    print(0)

    return


if __name__ == "__main__":
    main()
