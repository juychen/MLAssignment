import itertools
import math as m
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import PA2_IMP as im
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer


CLRS = ["r", "g", "b", "c","y","m","k"]
DATA = ['A', 'B', 'C']


def plot_cluster(X, Y, C, title='title'):
    for x, y, c in zip(X, Y, C):
        plt.scatter(x, y, color=CLRS[(c.astype(int))])

    plt.title(title)
    plt.axis([-15, 15, -15, 15])
    #plt.show()
    #plt.savefig(os.path.join('PA-2', 'plots', title + '.jpg'))
    plt.close()
    return


def plot_MS(X, Y, C, title='title'):
    for x, y, c in zip(X, Y, C):
        plt.scatter(x, y, c=[[c[0], c[1], 0]])

    plt.title(title)
    plt.axis([-15, 15, -15, 15])
    #plt.show()
    #plt.savefig(os.path.join('PA-2', 'plots', title + '.jpg'))

    plt.close()
    return


def main():

    exp_dict = {}

    for data in DATA:
        X = im.load_file(filename=os.path.join(
            'PA-2', 'data', 'cluster_data_data' + data + '_X.txt')).T
        exp_dict[(data, 'X')] = X

        Y = im.load_file(filename=os.path.join(
            'PA-2', 'data', 'cluster_data_data' + data + '_Y.txt')).T
        exp_dict[(data, 'Y')] = Y

        KM = im.Kmeans(k=4)
        KM.fit_x(X)
        KM.cluster()
        exp_dict[(data, 'KM')] = KM

        plot_cluster(X[:, 0], X[:, 1], KM.get_result(), 'data' + data + '_KM')

        GMM = im.EMGMM(k=4)
        GMM.fit_x(X)
        GMM.cluster()
        exp_dict[(data, 'GMM')] = GMM

        plot_cluster(X[:, 0], X[:, 1], GMM.get_result(),
                     'data' + data + '_GMM')

        # GMS = im.GaussianMeanShift(bandwidth=1)
        # GMS.fit_x(X)
        # GMS.cluster()
        # exp_dict[(data, 'GMS')] = GMS
        # scl = MinMaxScaler()
        # clrs = scl.fit_transform(np.around(GMS.x_,decimals=0))

        # plot_MS(X[:, 0], X[:, 1], clrs, 'data' + data + '_GMS')
    
    bandwidths = [1,3,10,20,50]

    for band in bandwidths:
        for data in DATA:
            X = exp_dict[(data, 'X')]
            Y = exp_dict[(data, 'Y')]
            GMS = im.GaussianMeanShift(bandwidth=band)
            GMS.fit_x(X)
            GMS.cluster()
            exp_dict[(data,band,'GMS')] = GMS
            xh = GMS.x_
            
            # KM2 = im.Kmeans(k=4)
            # KM2.fit_x(xh)
            # KM2.cluster()
            #plot_cluster(X[:, 0], X[:, 1], GMS.get_result(), 'data' + data + '_BD_'+ str(band)+'_GMS_KM')

            #print(GMS.get_result())

            scl = MinMaxScaler()
            clrs = scl.fit_transform(np.around(GMS.x_,decimals=0))
            #plot_MS(X[:, 0], X[:, 1], clrs, 'data' + data + '_BD_'+ str(band)+'_GMS')

    return


if __name__ == "__main__":
    main()
