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

METHODS = ['GMS', 'WGMS']
BANDS = [1, 2, 5, 7, 10]

def main():
    exp_dict = {}

    for data in DATA:

        img = Image.open(os.path.join(IMGPATH, data))
        pl.subplot(1, 3, 1)
        pl.imshow(img)

        X_raw, L = pa2.getfeatures(img, 7)
        X = vq.whiten(X_raw.T)

        exp_dict[(data, 'X')] = X

        for bd in BANDS:

            clf = im.GaussianMeanShift(itera=5,bandwidth=bd)
     
            clf.fit_x(X)
            clf.cluster()
            exp_dict[(data, 'MS', bd)] = clf

            Y = clf.get_result() + 1
            segm = pa2.labels2seg(Y, L)
            pl.subplot(1, 3, 2)
            pl.imshow(segm)
            csegm = pa2.colorsegms(segm, img)
            pl.subplot(1, 3, 3)
            pl.imshow(csegm)
            pl.savefig(os.path.join(OUTPATH, data + '_GMS_' + str(bd) + '_processed.jpg'))
            pl.show()

    for data in DATA:

        img = Image.open(os.path.join(IMGPATH, data))
        pl.subplot(1, 3, 1)
        pl.imshow(img)

        X_raw, L = pa2.getfeatures(img, 7)
        X = vq.whiten(X_raw.T)

        exp_dict[(data, 'X')] = X
        
        for bdp in BANDS:
            for bdc in BANDS:
                WKM = im.WeightGMeanshift(itera=5,chrominance_bandwidth=bdp,location_bandwidth=bdc)
                WKM.fit_x(X)
                WKM.cluster()
                exp_dict[(data, 'WGMS', bdp, bdc)] = WKM
                Y = WKM.get_result() + 1
                segm = pa2.labels2seg(Y, L)
                pl.subplot(1, 3, 2)
                pl.imshow(segm)
                csegm = pa2.colorsegms(segm, img)
                pl.subplot(1, 3, 3)
                pl.imshow(csegm)
                pl.savefig(os.path.join(OUTPATH, data + '_WGMS_' +
                                        str(bdp) + '_' + str(bdc) + '_processed.jpg'))
                pl.show()

        # KM = im.WeightGMeanshift(chrominance_bandwidth=4,location_bandwidth=10,itera=5)
        # KM.fit_x(X)
        # KM.cluster()
        # exp_dict[(data, 'KM')] = KM

        # np.savetxt(os.path.join(OUTPATH,data+'_meansh.txt'),KM.x_)

        # Y = KM.get_result() + 1
        # #C = KM.miu
        # print(np.unique(Y))

        # GMM = im.EMGMM(k=4)
        # # GMM.fit_x(X)
        # # GMM.cluster()
        # # exp_dict[(data, 'GMM')] = GMM
        # segm = pa2.labels2seg(Y,L)
        # pl.subplot(1,3,2)
        # pl.imshow(segm)

        # # color the segmentation image
        # csegm = pa2.colorsegms(segm, img)
        # pl.subplot(1,3,3)
        # pl.imshow(csegm)
        # pl.savefig(os.path.join(OUTPATH,data+'_processed.jpg'))
        # pl.show()

    return


if __name__ == "__main__":
    pl.switch_backend('agg')
    main()
