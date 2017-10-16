# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import math as m
import random
import os
import scipy as sc
from scipy.stats import poisson
from scipy.spatial.distance import euclidean



def poisson(x,Lambda):
    d = sc.stats.poisson.pmf(x,Lambda)
    return d 

def exp(data,K,L,itera=100,pij=[],threshold=0.001):
    Lambda = L
    

    N = len(data)
    Ns = np.zeros((N,K))
    
    if(len(pij)<1):
        pi = np.ones(K)/K
    else:
        pi = pij
    z = np.zeros((N,K)) 
    count = 0

    while (count<itera): 
        count=count+1
        # Q step
        for i in range(0,N):
            d = sum([pi[l] * poisson(data[i],Lambda[l]) for l in range(0,K)])
            for j in range(0,K):
                s = pi[j] * poisson(data[i],Lambda[j])          
                z[i,j] = s/d
        # P step
        Ns = np.sum(z,axis=0)

        pi_h = Ns/N
        Lambda_h = np.divide(np.dot(z.T,data),Ns)

        if(euclidean(pi_h,pi)<threshold and euclidean(Lambda,Lambda_h)<threshold):
            print(count)
            return Lambda_h,pi_h

        pi = pi_h      
        Lambda = Lambda_h

    print (itera)
    return Lambda,pi

def gendata(a):
    data = np.array([])

    for i in range(0,len(a)):
        data = np.append(data,np.ones(a[i])*i)
    
    return data

def main():
    ld = np.array([229,211,93,35,7,1])
    at = np.array([325,115,67,30,18,21])

    K = [1,2,3,4,5]

    ldata = gendata(ld)
    adata = gendata(at)

    result = {}

    print ('run')
    for k in K:
        lambda_h = [1+i*0.5 for i in range(0,k)]
        result[('london',k)] = exp(ldata,k,L=lambda_h)
        result[('anterwep',k)] = exp(adata,k,L=lambda_h)
    
    print (result)
    
if __name__ == "__main__":
    main()






