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

METHODS = ['WGMS']
BANDS = [ 3,4, 5 ,6,7]

def main():
    exp_dict = {}

    for data2 in DATA:

        img = Image.open(os.path.join(IMGPATH, data2))
        pl.subplot(1, 3, 1)
        pl.imshow(img)

        X_raw, L = pa2.getfeatures(img, 7)
        X = vq.whiten(X_raw.T)
        
        for bdp in BANDS:
            for bdc in BANDS:
                WKM = im.WeightGMeanshift(itera=5,chrominance_bandwidth=bdp,location_bandwidth=bdc)
                WKM.fit_x(X)
                WKM.cluster()
                Y = WKM.get_result() + 1
                segm = pa2.labels2seg(Y, L)
                pl.subplot(1, 3, 2)
                pl.imshow(segm)
                csegm = pa2.colorsegms(segm, img)
                pl.subplot(1, 3, 3)
                pl.imshow(csegm)
                pl.savefig(os.path.join(OUTPATH, data2 + '_WGMS_' +
                                        str(bdp) + '_' + str(bdc) + '_processed.jpg'))
                pl.show()
    return


if __name__ == "__main__":
    pl.switch_backend('agg')
    main()
