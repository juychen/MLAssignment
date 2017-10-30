import math as m
import os

import numpy as np
import pandas as pd
import PA2_IMP as im


def main():
    
    X_A = im.load_file(filename = os.path.join('PA-2','data','cluster_data_dataA_X.txt'))
    Y_A = im.load_file(filename = os.path.join('PA-2','data','cluster_data_dataA_Y.txt'))
    X_B = im.load_file(filename = os.path.join('PA-2','data','cluster_data_dataB_X.txt'))
    Y_B = im.load_file(filename = os.path.join('PA-2','data','cluster_data_dataB_Y.txt'))
    X_C = im.load_file(filename = os.path.join('PA-2','data','cluster_data_dataC_X.txt'))
    Y_C = im.load_file(filename = os.path.join('PA-2','data','cluster_data_dataC_Y.txt'))



    return


if __name__ == "__main__":
    main()
