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
BANDS = [ 4.375,  5.25 ,  6.125]

def main():
    exp_dict = {}

    for data in DATA:

        img = Image.open(os.path.join(IMGPATH, data))
        pl.subplot(1, 3, 1)
        pl.imshow(img)

        X_raw, L = pa2.getfeatures(img, 7)
        X = vq.whiten(X_raw.T)

        for bd in BANDS:

            clf = im.GaussianMeanShift(itera=5,bandwidth=bd)
     
            clf.fit_x(X)
            clf.cluster()

            Y = clf.get_result() + 1
            segm = pa2.labels2seg(Y, L)
            pl.subplot(1, 3, 2)
            pl.imshow(segm)
            csegm = pa2.colorsegms(segm, img)
            pl.subplot(1, 3, 3)
            pl.imshow(csegm)
            pl.savefig(os.path.join(OUTPATH, data + '_GMS_' + str(bd) + '_processed.jpg'))
            pl.show()
    return


if __name__ == "__main__":
    pl.switch_backend('agg')
    main()
