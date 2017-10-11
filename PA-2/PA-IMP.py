import math as m
import os

import numpy as np
import pandas as pd

from numpy.random import shuffle


class Kmeans:
    K = 1
    itera = 1 
    z = np.array([]) 
    x = np.array([])
    miu = np.array([])
    N = 0

    def __init__(self,k,itera):
        self.K = k
        self.itera = itera
    
    def initial_miu(self,pick_index=[]):
        if len(pick_index)==self.K:
            self.miu = self.x[pick_index]
            return 
        else :
            temp = self.x
            np.random.shuffle(temp)
            self.miu = temp[:self.K]
            return 
    
    def fit_x(self,x):
        self.x=x
        self.N=len(x)
        self.z = np.zeros((self.N,self.K))

        return 

    def cluster():

        return

print (0)

