import numpy as np
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import cvxopt
from scipy.stats import multivariate_normal
from cvxopt import matrix
from cvxopt import solvers

# cd D:\\OneDrive\\文档\\cityu\\MachineLearning\\MLAssignment\\PA-1\\PA-1-data-text

# define the polynomial function
def poly_function(x,order = 1):
    return np.array([m.pow(x,i) for i in range(0,order+1) ])

# load file from txt
def load_file(filename = 'polydata_data_polyx.txt'):
    return np.genfromtxt(filename,dtype='double')

def load_dataset():
    polyx = load_file(filename = 'polydata_data_polyx.txt')
    polyy = load_file(filename = 'polydata_data_polyy.txt')
    sampx = load_file(filename = 'polydata_data_sampx.txt')
    sampy = load_file(filename = 'polydata_data_sampy.txt')
    PHIX = PHIx(sampx,order=5,function='poly')

    return polyx,polyy,sampx,sampy,PHIX

# transpose 
def T(x):
    if(len(x.shape)>1):
        return np.transpose(x)
    else:
        return x.reshape(1,x.shape[0])

# x is a set of column vectors, get the transpose form of Φ matrix
def PHIx(x,order=1,function='poly'):
    if(function == 'poly'):
        mat = [poly_function(item,order) for item in x ]
        #return np.transpose(np.array(mat))
        return T(np.array(mat))

# return objective function according to different methods.
def obj_function(y,PHI,theta,Lambda=0,method='LS'):
    if method == 'LS':
        return np.linalg.norm(y-np.dot(T(PHI),theta),ord=2)
    if method == 'RLS':
        return np.linalg.norm(y-np.dot(T(PHI),theta),ord=2) + Lambda * np.linalg.norm(theta,ord=2)
    if method == 'LASSO':
        return np.linalg.norm(y-np.dot(T(PHI),theta),ord=2) + Lambda * np.linalg.norm(theta,ord=1)
    if method == 'RR':
        return np.linalg.norm(y-np.dot(T(PHI),theta),ord=1)

# Generate prediction according to the theta
def predict(x,theta,function='poly'):
    if(function=='poly'):
        PHIX=PHIx(x,order=theta.shape[0]-1,function=function)
        predections = np.dot(T(PHIX),theta)
        return predections

# parameter estimate , all input vectors are column vectors
def para_estimate(y,PHI,Lambda=0.1,method='LS'):
    if method == 'LS':
        return np.dot(np.dot(np.linalg.inv(np.dot(PHI,T(PHI))),PHI),y)
    if method == 'RLS':
        return np.dot(np.dot(np.linalg.inv(np.dot(PHI,T(PHI))+Lambda*np.eye(PHI.shape[0])),PHI),y)
    if method == 'LASSO':
        PHIPHIT = np.dot(PHI,T(PHI))
        PHIy = np.dot(PHI,y)

        H = np.vstack((np.hstack((PHIPHIT,-1*PHIPHIT)),
                       np.hstack((-1*PHIPHIT,PHIPHIT))))
        
        f = np.hstack((PHIy,-1*PHIy))
        f = Lambda * np.ones(f.shape) - f

        P = matrix(H)
        q = matrix(f)
        G = matrix(np.eye(len(f))*-1)
        h = matrix(np.zeros(len(f)))
        
        sol = solvers.qp(P,q,G,h)
        x = sol['x']
        theta = x[:int(len(x)/2)]- x[int(len(x)/2):]

        return np.array(theta)
    if method == 'RR':

        A = np.vstack((np.hstack((-1*T(PHI),-1*np.eye(T(PHI).shape[0]))),
                       np.hstack((T(PHI),-1*np.eye(T(PHI).shape[0])))))
        b = np.hstack((-1*y,
                       y))

        f = np.hstack((np.zeros(T(PHI).shape[1]),
                       np.ones(T(PHI).shape[0])))

        c = matrix(f)
        A = matrix(A)
        b = matrix(b)

        sol = solvers.lp(c,A,b)

        return 

# define posterior of Bayesian Regression
def posterior_BR(x,y,PHI,alpha=0.1,sigma=0.1):
    SIGMA_theta = np.linalg.inv(1/alpha*np.eye(PHI.shape[0])+1/(sigma*sigma)*np.dot(PHI,T(PHI)))
    miu_theta = 1/(sigma*sigma)*np.dot(np.dot(SIGMA_theta,PHI),y) 
    #posterior = multivariate_normal(x,miu_theta,SIGMA_theta)
    return miu_theta,SIGMA_theta

def predict_BR(x,miu_theta,SIGMA_theta,function='poly'):

    if(function=='poly'):
        PHIX = PHIx(x,order=miu_theta.shape[0]-1,function=function)
        miu_star = np.dot(T(PHIX),miu_theta)
        sigma_theta_sqr = np.dot(np.dot(T(PHIX),SIGMA_theta),PHIX)
        return miu_star,sigma_theta_sqr

# Generate Plots with
def plot_f_s(x,y,sampx,sampy,label):
    plt.plot(x, y, label=label)
    plt.legend()
    plt.plot(sampx, sampy,'ro',label='data')
    plt.legend()
    plt.show()
    return 

def plot_f_s_std(x,y,sampx,sampy,deviation,label):
    plt.plot(x, y, label=label)
    plt.legend()
    plt.plot(sampx, sampy,'ro',label='data')
    plt.legend()
    plt.errorbar(x, y, deviation, linestyle='None', marker='^')
    plt.show()
    return

def main():
    #polyx = load_file(filename = 'polydata_data_polyx.txt')
    #polyy = load_file(filename = 'polydata_data_polyy.txt')
    #sampx = load_file(filename = 'polydata_data_sampx.txt')
    #sampy = load_file(filename = 'polydata_data_sampy.txt')

    #PHIX = PHIx(sampx,order=5,function='poly')

    polyx,polyy,sampx,sampy,PHIX = load_dataset()

    theta_LS = para_estimate(sampy,PHIX,method='LS')
    prediction_LS = predict(polyx,theta_LS,function='poly')
    plot_f_s(polyx,prediction_LS,sampx,sampy,label='Least-squares Regression')

    theta_RLS = para_estimate(sampy,PHIX,Lambda=0.1,method='RLS')
    prediction_RLS = predict(polyx,theta_RLS,function='poly')
    plot_f_s(polyx,prediction_RLS,sampx,sampy,label='Regularize LS Regression')

    theta_LASSO = para_estimate(sampy,PHIX,Lambda=0.1,method='LASSO')
    prediction_LASSO = predict(polyx,theta_LASSO,function='poly')
    plot_f_s(polyx,prediction_LASSO,sampx,sampy,label='Regularize LASSO Regression')

    miu_theta,SIGMA_theta = posterior_BR(sampx,sampy,PHIX)
    miu_star,sigma_thea_sqr = predict_BR(polyx,miu_theta,SIGMA_theta)
    plot_f_s_std(polyx,miu_star,sampx,sampy,np.sqrt(sigma_thea_sqr.diagonal()),label='Bayesian Regression')

    # my code here

if __name__ == "__main__":
    main()



