import math as m
import os

import numpy as np
import pandas as pd
import pylab as pl
import scipy.io as sio
from PIL import Image  # python3

import pa2
import PA2_IMP as im
import scipy.cluster.vq as vq


IMGPATH = os.path.join('PA-2', 'data', 'images')
OUTPATH = os.path.join('PA-2', 'data', 'processed')

DATA = os.listdir(IMGPATH)


def main():
    exp_dict = {}

    for data in DATA[:1]:

        img = Image.open(os.path.join(IMGPATH, data))
        pl.subplot(2, 3, 1)
        pl.imshow(img)

        X_raw, L = pa2.getfeatures(img, 7)
        X = vq.whiten(X_raw.T)

        # X = im.load_file(filename=os.path.join(
        #     'PA-2', 'data', 'cluster_data_data' + data + '_X.txt')).T
        exp_dict[(data, 'X')] = X

        KM = im.GaussianMeanShift(bandwidth=2,tolarance=1)
        KM.fit_x(X)
        KM.cluster()
        exp_dict[(data, 'KM')] = KM

        Y = KM.get_result() + 1
        #C = KM.miu
        print(Y)

        # GMM = im.EMGMM(k=4)
        # GMM.fit_x(X)
        # GMM.cluster()
        # exp_dict[(data, 'GMM')] = GMM
        segm = pa2.labels2seg(Y,L)
        pl.subplot(2,3,2)
        pl.imshow(segm)
    
        # color the segmentation image
        csegm = pa2.colorsegms(segm, img)
        pl.subplot(2,3,3)
        pl.imshow(csegm)
        pl.savefig(os.path.join(OUTPATH,data+'_processed.jpg'))
        pl.show()


    return


if __name__ == "__main__":
    pl.switch_backend('agg')
    main()
