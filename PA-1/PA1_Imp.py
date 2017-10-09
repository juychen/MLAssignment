import numpy as np
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import cvxopt
import os
from scipy.stats import multivariate_normal
from sklearn.utils import resample
from cvxopt import matrix
from cvxopt import solvers

import matplotlib

# cd D:\\OneDrive\\文档\\cityu\\MachineLearning\\MLAssignment\\

NAME_MAP = {'LS':'Least Square Regression',
            'RLS':'Regularized LS',
            'LASSO':'L1-Regularized LS',
            'RR':'Robust Regression',
            'BR':'Bayesian Regression'}

# define the polynomial function
def poly_function(x,order = 1):
    if(type(x) is not list and type(x) is not np.ndarray):
        return np.array([pow(x,i) for i in range(0,order+1) ])
    else:
        row_vect = np.array([])
        row_vect= np.append(row_vect,[1])
        for item in x:
            row_vect = np.append(row_vect,np.array([pow(item,i) for i in range(1,order+1) ]))
        return row_vect
# define the cross term 2 order 
def cross_term_function (x):

    row_vect = np.array([])
    row_vect= np.append(row_vect,[1])
    
    tempx = np.array(x)
    m = np.dot(T(tempx).T,T(tempx))
    upper_index = np.triu_indices(m.shape[1])
    row_vect= np.append(row_vect,m[upper_index])
    #newx = row_vect[1:]

    return row_vect

# load file from txt
def load_file(filename = 'polydata_data_polyx.txt'):
    return np.genfromtxt(filename,dtype='double')

def load_dataset():
    polyx = load_file(filename = os.path.join('PA-1','PA-1-data-text','polydata_data_polyx.txt'))
    polyy = load_file(filename = os.path.join('PA-1','PA-1-data-text','polydata_data_polyy.txt'))
    sampx = load_file(filename = os.path.join('PA-1','PA-1-data-text','polydata_data_sampx.txt'))
    sampy = load_file(filename = os.path.join('PA-1','PA-1-data-text','polydata_data_sampy.txt'))

    return polyx,polyy,sampx,sampy

def load_dataset_P2():
    polyx = load_file(filename = os.path.join('PA-1','PA-1-data-text','count_data_testx.txt'))
    polyy = load_file(filename = os.path.join('PA-1','PA-1-data-text','count_data_testy.txt'))
    sampx = load_file(filename = os.path.join('PA-1','PA-1-data-text','count_data_trainx.txt'))
    sampy = load_file(filename = os.path.join('PA-1','PA-1-data-text','count_data_trainy.txt'))

    return polyx,polyy,sampx,sampy

# transpose 
def T(x):
    if(len(x.shape)>1):
        return np.transpose(x)
    else:
        return x.reshape(x.shape[0],1)

# x is a set of column vectors, get the transpose form of Φ matrix
def PHIx(x,order=5,function='poly'):
    if(function == 'id'):
        return x
    if(function == 'poly'):
        if(len(x.shape)<2):
            mat = [poly_function(item,order) for item in x ]
        else:
            mat = [poly_function(item,order) for item in x.T ]
        return T(np.array(mat))
    if(function == 'corss'):

        mat = [cross_term_function(item) for item in x.T ]
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
    if(function=='poly' or function == 'id'):
        PHIX=PHIx(x,order=int((theta.shape[0]-1)/T(x).T.shape[0]),function=function)
        predections = np.dot(T(PHIX),theta)
        return predections
    if (function=='cross'):
        PHIX = PHIx(x,function=function)
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
def mse(y,prediction):

    if(len(y)!=len(prediction)):
        return m.inf

    ry =  y.reshape(len(y),1)
    rp = prediction.reshape(len(prediction),1)
    e = ry - rp
    return (np.dot(T(e),e)/len(e))[0,0]

# define mean absolute error
def mae(y,prediction):

    if(len(y)!=len(prediction)):
        return m.inf

    ry =  y.reshape(len(y),1)
    rp = prediction.reshape(len(prediction),1)
    e = ry - rp
    return np.linalg.norm(e,ord=1)/len(e)

# define posterior of Bayesian Regression
def posterior_BR(x,y,PHI,alpha=0.1,sigma=0.1):
    SIGMA_theta = np.linalg.inv(1/alpha*np.eye(PHI.shape[0])+1/(sigma*sigma)*np.dot(PHI,T(PHI)))
    miu_theta = 1/(sigma*sigma)*np.dot(np.dot(SIGMA_theta,PHI),y) 
    #posterior = multivariate_normal(x,miu_theta,SIGMA_theta)
    return miu_theta,SIGMA_theta

# define predictive model of Bayesan Regression
def predict_BR(x,miu_theta,SIGMA_theta,function='poly'):

    if(function=='poly' or function=='id'):
        PHIX = PHIx(x,order=int((miu_theta.shape[0]-1)/T(x).T.shape[0]),function=function)
        miu_star = np.dot(T(PHIX),miu_theta)
        sigma_theta_sqr = np.dot(np.dot(T(PHIX),SIGMA_theta),PHIX)
        return miu_star,sigma_theta_sqr
    
    if (function=='cross'):
        PHIX = PHIx(x,function=function)
        miu_star = np.dot(T(PHIX),miu_theta)
        sigma_theta_sqr = np.dot(np.dot(T(PHIX),SIGMA_theta),PHIX)
        return miu_star,sigma_theta_sqr

# Generate Plots 
def plot_f_s(x,y,pred,sampx,sampy,label):

    plt.plot(x, y, label='True Function',c='k')
    plt.legend()
    plt.plot(x, pred, label=label,c='b')
    plt.legend()
    plt.plot(sampx, sampy,'ro',label='data')
    plt.legend()
    plt.savefig(os.path.join('PA-1','plots',label+'.jpg'))
    plt.close()
    return 

def plot_f_s_std(x,y,pred,sampx,sampy,deviation,label):

    plt.plot(x, y, label='True Function',c='k')
    plt.legend()
    plt.plot(x, pred, label=label,c='b')  
    plt.legend()  
    plt.plot(sampx, sampy,'ro',label='data')
    plt.legend()
    plt.errorbar(x, pred,yerr = deviation)
    plt.savefig(os.path.join('PA-1','plots',label+'.jpg'))
    plt.close()

    return

# model selection to search the best parameter 
def model_selection(polyx,polyy,sampx,sampy,param_dict,estimator='RLS'):
    para_err_map={}
    best_para={}

    if (estimator == 'RLS' or estimator == 'LASSO'):
        Lambdas = param_dict['Lambda']
        functions = param_dict['function']
        orders = param_dict['order']

        for order in orders:
            for function in functions:
                PHIX = PHIx(sampx,order=order,function=function)
                for Lambda in Lambdas:
                    theta = para_estimate(sampy,PHIX,Lambda=Lambda,method=estimator)
                    prediction = predict(polyx,theta,function=function)
                    err = mse(prediction,polyy)
                    paraset = {'function':function,'order':order,'Lambda':Lambda}
                    para_err_map[str(paraset)] = err
    
    if (estimator == 'BR'):
        alphas = param_dict['alpha']
        sigmas = param_dict['sigma']
        functions = param_dict['function']
        orders = param_dict['order']

        for order in orders:
            for function in functions:
                PHIX = PHIx(sampx,order=order,function=function)
                for alpha in alphas:
                    for sigma in sigmas:
                        theta,SIGMA_theta = posterior_BR(sampx,sampy,PHIX,alpha=alpha,sigma=sigma)
                        prediction,cov = predict_BR(polyx,theta,SIGMA_theta,function=function)
                        err = mse(prediction,polyy)
                        paraset = {'function':function,'order':order,'alpha':alpha,'sigma':sigma}
                        para_err_map[str(paraset)] = err        
    
    best = min(para_err_map, key=para_err_map.get)
    best_para = eval(best)
    return para_err_map,best_para

# Plot learning curve with different data size
def learning_curve(polyx,polyy,sampx,sampy,paradict={},subset=[1],repeat=1,method='LS',plot_title='Learning Curve LS',show_plot=True,ylim=0):
    err = []
    if(show_plot==True): plt.plot(polyx, polyy, label='True Function',c='k')
    for size in subset:
        nsamp = int(size*len(sampy))
        err_perround = 0
        for i in range(0,repeat):
            if(len(sampx.shape)<2):
                resampx, resampy = resample(sampx, sampy,n_samples=nsamp,replace=False, random_state=i*17)
            else :
                resampx, resampy = resample(sampx.T, sampy,n_samples=nsamp,replace=False, random_state=i*17)
                resampx = resampx.T

            round_err,prediction = experiment(polyx,polyy,resampx,resampy,paradict,method=method,plot_title=NAME_MAP[method]+' subset '+str(round(size,1)),show_plot=False)
            if(i==0 and show_plot == True):
                plt.legend()
                plt.plot(polyx, prediction, label=NAME_MAP[method]+' subset '+str(round(size,1)))
                plt.legend()

            # if parameter dictionnary is not empty
            #if method == 'BR':
                #theta,SIGMA_theta, prediction,cov = experiment(polyx,polyy,resampx,resampy,paradict,method=method,plot_title=method+' '+str(size))
            #else :
                #theta, prediction = experiment(polyx,polyy,resampx,resampy,paradict,method=method,plot_title=method+' '+str(size))
            err_perround += round_err
        if(show_plot == True):
            plt.savefig(os.path.join('PA-1','plots',NAME_MAP[method]+' in '+str(repeat)+' rounds.jpg'))
        
        err.append(err_perround/repeat)

    if(show_plot == True): plt.close()        
    plt.plot(subset, err, label=plot_title,c='b')
    if(ylim !=0):
        plt.ylim((0,ylim))
    plt.legend()
    plt.savefig(os.path.join('PA-1','plots',plot_title+'.jpg'))
    plt.close()

    return err

# Define experiment with certain method and data
def experiment(polyx,polyy,sampx,sampy,paradict={},method='LS',plot_title='Least-squares Regression',show_plot=True,save_theta=False):
    
    prediction= np.array([])
    theta = np.array([])

    # parameter is not empty
    if paradict:

        try:
            if (method == 'BR'):
                PHIX = PHIx(sampx,order=paradict['order'],function=paradict['function'])
                theta,SIGMA_theta = posterior_BR(sampx,sampy,PHIX,alpha=paradict['alpha'],sigma=paradict['sigma'])
                prediction,cov = predict_BR(polyx,theta,SIGMA_theta,function=paradict['function'])
                if(show_plot==True):
                    plot_f_s_std(polyx,polyy,prediction,sampx,sampy,np.sqrt(np.sqrt(cov.diagonal())),label=plot_title)
                #return theta,SIGMA_theta, prediction,cov
                if(save_theta == True):
                    thetaF = pd.DataFrame(theta)
                    thetaF.to_csv('theta_'+plot_title+'.csv')
                return mse(prediction,polyy),prediction


            else:
                PHIX = PHIx(sampx,order=paradict['order'],function=paradict['function'])
                theta = para_estimate(sampy,PHIX,Lambda=paradict['Lambda'],method=method)
                prediction = predict(polyx,theta,function=paradict['function'])
                if(show_plot==True):
                    plot_f_s(polyx,polyy,prediction,sampx,sampy,label=plot_title) 
                #return theta, prediction
                if(save_theta == True):
                    thetaF = pd.DataFrame(theta)
                    thetaF.to_csv('theta_'+plot_title+'.csv')
                return mse(prediction,polyy),prediction
            
        except Exception as e:
            print ('missing parameter: ')
            print (e)
            return 

    # parameter is empty
    if (method == 'BR'):
            PHIX = PHIx(sampx)
            theta,SIGMA_theta = posterior_BR(sampx,sampy,PHIX)
            prediction,cov = predict_BR(polyx,theta,SIGMA_theta)
            if(show_plot==True):
                plot_f_s_std(polyx,polyy,prediction,sampx,sampy,np.sqrt(np.sqrt(cov.diagonal())),label=plot_title)
            #return theta,SIGMA_theta, prediction,cov
            if(save_theta == True):
                thetaF = pd.DataFrame(theta)
                thetaF.to_csv('theta_'+plot_title+'.csv')
            return mse(prediction,polyy),prediction

    PHIX = PHIx(sampx)
    theta = para_estimate(sampy,PHIX,method=method)
    prediction = predict(polyx,theta)
    if(show_plot==True):
        plot_f_s(polyx,polyy,prediction,sampx,sampy,label=plot_title)
    
    if(save_theta == True):
        thetaF = pd.DataFrame(theta)
        thetaF.to_csv('theta_'+plot_title+'.csv')
    #return theta, prediction
    return mse(prediction,polyy),prediction

# Set experiments with outliers
def outliers_experiments(polyx,polyy,sampx,sampy,olx,oly,paradict={},method='LS',plot_title='Least-squares Regression'):

    addedx = np.hstack((sampx,olx))
    addedy = np.hstack((sampy,oly))

    return experiment(polyx,polyy,addedx,addedy,paradict=paradict,method=method,plot_title=plot_title)

def mseMap_toCSV(msedict,fname='mse.csv'):

    if(str(type(msedict))!="<class 'dict'>"):
        mseMap_toCSV({'nohparam':str(msedict)},fname)
        return 

    keys = list(msedict.keys())
    values = list(msedict.values())
    mseDf = pd.DataFrame({'mse':values},index=keys)
    mseDf.sort_values('mse',ascending=True,inplace=True)
    mseDf.to_csv(os.path.join('PA-1','plots',fname))
    return