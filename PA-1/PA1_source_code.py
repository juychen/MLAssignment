import numpy as np
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import cvxopt
from scipy.stats import multivariate_normal
from sklearn.utils import resample
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
def PHIx(x,order=5,function='poly'):
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

        theta = np.array(sol['x'][:T(PHI).shape[1]])
        return theta

# define mean square error
def mse(y,predction):

    if(len(y)!=len(predction)):
        return m.inf

    ry =  y.reshape(len(y),1)
    rp = predction.reshape(len(predction),1)
    e = ry - rp
    return (np.dot(T(e),e)/len(e))[0,0]

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
def plot_f_s(x,y,pred,sampx,sampy,label):
    plt.plot(x, y, label='True Function',c='k')
    plt.legend()
    plt.plot(x, pred, label=label,c='b')
    plt.legend()
    plt.plot(sampx, sampy,'ro',label='data')
    plt.legend()
    plt.show()
    return 

# model_selection
def model_selection(polyx,polyy,sampx,sampy,param_dict,estimator='RLS'):
    opt_para={}
    min_err = m.inf

    if (estimator == 'RLS' or estimator == 'LASSO'):
        Lambdas = param_dict['Lambda']
        functions = param_dict['function']

        for function in functions:
            
            PHIX = PHIx(sampx,order=param_dict['order'],function=param_dict['function'])

            for Lambda in Lambdas:
                    theta = para_estimate(sampy,PHIX,Lambda=Lambda,method=estimator)
                    prediction = predict(polyx,theta,function=function)
                    err = mse(prediction,polyy)
                    opt_para[function,Lambda] = err
    
    if (estimator == 'BR'):
        alphas = param_dict['alpha']
        sigmas = param_dict['sigma']
        functions = param_dict['function']

        for function in functions:
            PHIX = PHIx(sampx,order=param_dict['order'],function=param_dict['function'])

            for alpha in alphas:
                for sigma in sigmas:
                     theta,SIGMA_theta = posterior_BR(sampx,sampy,PHIX,alpha=alpha,sigma=sigma)
                     prediction,cov = predict_BR(polyx,theta,SIGMA_theta,function=function)
                     err = mse(prediction,polyy)
                     opt_para[function,alpha,sigma] = err
    
    best = min(opt_para, key=d.get)
    return opt_para,best

def plot_f_s_std(x,y,pred,sampx,sampy,deviation,label):
    plt.plot(x, y, label='True Function',c='k')
    plt.legend()
    plt.plot(x, pred, label=label,c='b')  
    plt.legend()  
    plt.plot(sampx, sampy,'ro',label='data')
    plt.legend()
    plt.errorbar(x, pred,yerr = deviation)
    plt.show()
    return

def learning_curve(polyx,polyy,sampx,sampy,paradict={},subset=[1],repeat=1,method='LS',plot_title='Learning Curve'):
    err = []
    for size in subset:
        nsamp = int(size*len(sampy))
        err_perround = 0
        for i in range(0,repeat):
            resampx, resampy = resample(sampx, sampy,n_samples=nsamp,replace=False, random_state=i)
            # if parameter dictionnary is not empty
            if method == 'BR':
                theta,SIGMA_theta, prediction,cov = experiment(polyx,polyy,resampx,resampy,paradict,method=method,plot_title=method+' '+str(size))
            else :
                theta, prediction = experiment(polyx,polyy,resampx,resampy,paradict,method=method,plot_title=method+' '+str(size))
            err_perround += mse(polyy,prediction)
                
        err.append(err_perround/repeat)

    plt.plot(subset, err, label='Leaning Curve',c='b')
    plt.legend()
    plt.show()

    return err

def experiment(polyx,polyy,sampx,sampy,paradict={},method='LS',plot_title='Least-squares Regression'):
    
    prediction= np.array([])
    theta = np.array([])

    # parameter is not empty
    if paradict:

        try:
            if (method == 'BR'):
                PHIX = PHIx(sampx,order=paradict['order'],function=paradict['function'])
                theta,SIGMA_theta = posterior_BR(sampx,sampy,PHIX,alpha=paradict['alpha'],sigma=paradict['sigma'])
                prediction,cov = predict_BR(polyx,theta,SIGMA_theta,function=paradict['function'])
                plot_f_s_std(polyx,polyy,prediction,sampx,sampy,np.sqrt(np.sqrt(cov.diagonal())),label=plot_title)
                return theta,SIGMA_theta, prediction,cov


            else:
                PHIX = PHIx(sampx,order=paradict['order'],function=paradict['function'])
                theta = para_estimate(sampy,PHIX,Lambda=paradict['Lambda'],method=method)
                prediction = predict(polyx,theta,function=paradict['function'])
                plot_f_s(polyx,polyy,prediction,sampx,sampy,label=plot_title) 
                return theta, prediction
            
        except Exception as e:
            print ('missing parameter: ')
            print (e)
            return 

    # parameter is empty
    if (method == 'BR'):
            PHIX = PHIx(sampx)
            theta,SIGMA_theta = posterior_BR(sampx,sampy,PHIX)
            prediction,cov = predict_BR(polyx,theta,SIGMA_theta)
            plot_f_s_std(polyx,polyy,prediction,sampx,sampy,np.sqrt(np.sqrt(cov.diagonal())),label=plot_title)
            return theta,SIGMA_theta, prediction,cov

    PHIX = PHIx(sampx)
    theta = para_estimate(sampy,PHIX,method=method)
    prediction = predict(polyx,theta)
    plot_f_s(polyx,polyy,prediction,sampx,sampy,label=plot_title)

    return theta, prediction

def outliers_experiments(polyx,polyy,sampx,sampy,olx,oly,paradict={},method='LS',plot_title='Least-squares Regression'):

    addedx = np.hstack((sampx,olx))
    addedy = np.hstack((sampy,oly))

    return experiment(polyx,polyy,addedx,addedy,paradict=paradict,method=method,plot_title=plot_title)

def main():

    polyx,polyy,sampx,sampy,PHIX = load_dataset()

    experiment(polyx,polyy,sampx,sampy,method='LS',plot_title='Least-squares Regression')

    experiment(polyx,polyy,sampx,sampy,method='RLS',plot_title='Regularize LS Regression')

    experiment(polyx,polyy,sampx,sampy,method='LASSO',plot_title='Regularize LASSO Regression')

    experiment(polyx,polyy,sampx,sampy,method='RR',plot_title='Robust Regression')

    experiment(polyx,polyy,sampx,sampy,method='BR',plot_title='Bayesian Regression')
    # my code here

if __name__ == "__main__":
    main()