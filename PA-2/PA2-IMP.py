import math as m
import os

import numpy as np
import pandas as pd
from numpy.random import shuffle
from scipy.spatial.distance import euclidean
from scipy.stats import multivariate_normal

from sklearn.utils import resample

def gaussian(x,miu,SIGMA):
    return multivariate_normal.pdf(x,miu,SIGMA)

# transpose 
def T(x):
    if(str(type(x))!="<class 'numpy.ndarray'>"):
        x = np.array(x)
    if(len(x.shape)>1):
        return np.transpose(x)
    else:
        return x.reshape(x.shape[0],1)

class ClusterAlgorithm:
    K = 1
    itera = 1 
    z = np.array([]) 
    x = np.array([])
    threshold = 0
    N = 0

    def __init__(self,k=1,itera=10,threshold=0.005):
        self.K = k
        self.itera = itera
        self.threshold = threshold

    def fit_x(self,x):
        self.x=x
        self.N=len(x)
        self.z = np.zeros((self.N,self.K))
        return 

class EMMM(ClusterAlgorithm):

    pi = np.array([])

    def fit_x(self,x,pi=[]):
        ClusterAlgorithm.fit_x(self,x)
        self.initial_pi(pi)
        return 
    
    def initial_pi(self,pi_init=[]):
        if(len(pi_init)==self.K):
            self.pi = np.array(pi_init)
            return 
        else:
            self.pi = np.ones(self.K)/self.K
            return 

class EMGMM(EMMM):

    miu = np.array([])
    SIGMA = np.array([])
    d = 0

    def __init__(self,k=1,itera=10,threshold=0.005,dimension=1):
        EMMM.__init__(self,k,itera,threshold)
        self.d = dimension

    def fit_x(self,x,miu=[],SIGMA=np.array([])):
        EMMM.fit_x(self,x)
        initial_miu(miu_init=miu)
        initial_SIGMA(SIGMA_init=SIGMA)

    def initial_miu(self,miu_init=[]):
        if(len(pi_init)==self.d):
            self.miu = np.array(miu_init)
        else:
            self.miu = np.linspace(1.0,2.0,num=self.d)
        return

    def initial_SIGMA(self,SIGMA_init=np.array([])):
        if(SIGMA_init.shape ==(self.d,self.d)):
            self.SIGMA = SIGMA_init
        else:
            self.SIGMA = np.eye(self.d)
        return 
         
class Kmeans(ClusterAlgorithm):

    miu = np.array([])
  
    def initial_miu(self,pick_index=[]):
        if (len(pick_index)==self.K):
            self.miu = self.x[pick_index]
            return 
        else :
            self.miu = resample(self.x,n_samples=self.K,replace=False)
            return 
    
    def fit_x(self,x,pick=[]):
        ClusterAlgorithm.fit_x(self,x)
        self.initial_miu(pick)
        return 

    def cluster(self):

        if(len(self.x)<1): 
            print ('no data')
            return 

        count = 0
        
        while(count<self.itera):

            z = np.array([])
            count+=1
            distance = np.array([np.linalg.norm(self.x - item,axis=1) for item in self.miu]).T
            
            for item in distance:
               midx = np.argmin(item)
               item = 0 * item
               item[midx] = 1
               z = np.append(z,item)
            
            self.z = z.reshape(distance.shape)

            tmiu = (np.dot(self.z.T,self.x)) / np.sum(self.z,axis=0)[:,None]

            if(np.linalg.norm(self.miu-tmiu)<self.threshold):
                self.miu = tmiu
                return

            else : 
                self.miu = tmiu
        return

def main():

     a = Kmeans(k=2,itera=10)
     a.fit_x (np.array([[1,2,3],[1,5,3],[2,4,1]]),pick=[1,2])
     #a.initial_miu(pick_index=[0,1])
     print(a.miu)
     a.cluster()

     print (a.z)
     print (a.miu)
     return 


if __name__ =="__main__":
    main()

print (0)
