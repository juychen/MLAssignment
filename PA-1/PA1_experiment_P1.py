import PA1_Imp as imp
import numpy as np
import pandas as pd
import math as m
import os

NAME_MAP = imp.NAME_MAP

def main():

    polyx,polyy,sampx,sampy = imp.load_dataset()
    
    # methods without hyper parameters
    mse_LS = imp.experiment(polyx,polyy,sampx,sampy,method='LS',plot_title=NAME_MAP['LS'])
    imp.mseMap_toCSV({'nohparam':mse_LS},'mse_LS.csv')

    mse_RR = imp.experiment(polyx,polyy,sampx,sampy,method='RR',plot_title=NAME_MAP['RR'])
    imp.mseMap_toCSV({'nohparam':mse_RR},'mse_RR.csv')


    # methods with hyper parameters
    para_RLS = {'Lambda':[0.1,0.25,0.5,1,2,5],'function':['poly'],'order':[5]}
    para_err_RLS,opt_para_RLS = imp.model_selection(polyx,polyy,sampx,sampy,para_RLS,estimator='RLS')
    imp.mseMap_toCSV(para_err_RLS,'mse_RLS.csv')
    mse_RLS = imp.experiment(polyx,polyy,sampx,sampy,paradict=opt_para_RLS,method='RLS',plot_title=NAME_MAP['RLS'])

    para_LASSO = {'Lambda':[0.1,0.25,0.5,1,2,5],'function':['poly'],'order':[5]}
    para_err_LASSO,opt_para_LASSO = imp.model_selection(polyx,polyy,sampx,sampy,para_LASSO,estimator='LASSO')
    imp.mseMap_toCSV(para_err_LASSO,'mse_LASSO.csv')
    mse_LASSO = imp.experiment(polyx,polyy,sampx,sampy,paradict=opt_para_LASSO,method='LASSO',plot_title=NAME_MAP['LASSO'])

    para_BR = {'alpha':[0.1,0.5,1,5],'sigma':[0.1,0.5,1,5],'function':['poly'],'order':[5]}
    para_err_BR,opt_para_BR = imp.model_selection(polyx,polyy,sampx,sampy,para_BR,estimator='BR')
    imp.mseMap_toCSV(para_err_BR,'mse_BR.csv')
    mse_BR = imp.experiment(polyx,polyy,sampx,sampy,paradict=opt_para_BR,method='BR',plot_title=NAME_MAP['BR'])

    # learning curve imp.experiments
    subset = np.linspace(0.2,1,5)
    err_LS = imp.learning_curve(polyx,polyy,sampx,sampy,subset=subset,repeat=10,method='LS',plot_title='Learning Curve '+NAME_MAP['LS'])
    err_RLS = imp.learning_curve(polyx,polyy,sampx,sampy,subset=subset,repeat=10,method='RLS',plot_title='Learning Curve '+NAME_MAP['RLS'])
    err_LASSO = imp.learning_curve(polyx,polyy,sampx,sampy,subset=subset,repeat=10,method='LASSO',plot_title='Learning Curve '+NAME_MAP['LASSO'])
    err_RR = imp.learning_curve(polyx,polyy,sampx,sampy,subset=subset,repeat=10,method='RR',plot_title='Learning Curve '+NAME_MAP['RR'])
    err_BR = imp.learning_curve(polyx,polyy,sampx,sampy,subset=subset,repeat=10,method='BR',plot_title='Learning Curve '+NAME_MAP['BR'])

    # outliers imp.experiments
    outliers_x = [-1.3,0.5,0.7,1]
    outliers_y = [80,30,50,-30]

    mseol_LS = imp.outliers_experiments(polyx,polyy,sampx,sampy,outliers_x,outliers_y,method='LS',plot_title=NAME_MAP['LS']+' with Outliers')
    mseol_RR = imp.outliers_experiments(polyx,polyy,sampx,sampy,outliers_x,outliers_y,method='RR',plot_title=NAME_MAP['RR']+' with Outliers')
    mseol_RLS = imp.outliers_experiments(polyx,polyy,sampx,sampy,outliers_x,outliers_y,paradict=opt_para_RLS,method='RLS',plot_title=NAME_MAP['RLS']+' with Outliers')
    mseol_LASSO = imp.outliers_experiments(polyx,polyy,sampx,sampy,outliers_x,outliers_y,paradict=opt_para_LASSO,method='LASSO',plot_title=NAME_MAP['LASSO']+' with Outliers')
    mseol_BR = imp.outliers_experiments(polyx,polyy,sampx,sampy,outliers_x,outliers_y,paradict=opt_para_BR,method='BR',plot_title=NAME_MAP['BR']+' with Outliers')

    # higer order imp.experiments
    para_Lambda_o10 = {'Lambda':[0.1,0.25,0.5,1,2,5],'function':['poly'],'order':[10]}
    para_BR_o10 = {'alpha':[0.1,0.5,1,5],'sigma':[0.1,0.5,1,5],'function':['poly'],'order':[10]}

    mse_LS_o10 = imp.experiment(polyx,polyy,sampx,sampy,paradict={'function':'poly','order':10,'Lambda':0},method='LS',plot_title=NAME_MAP['LS']+' order 10')
    imp.mseMap_toCSV({'nohparam':mse_LS_o10},'mse_LS_o10.csv')

    mse_RR_o10 = imp.experiment(polyx,polyy,sampx,sampy,paradict={'function':'poly','order':10,'Lambda':0},method='RR',plot_title=NAME_MAP['RR']+' order 10')
    imp.mseMap_toCSV({'nohparam':mse_RR_o10},'mse_RR_o10.csv')

    # methods with hyper parameters
    para_err_RLS_o10,opt_para_RLS_o10 = imp.model_selection(polyx,polyy,sampx,sampy,para_Lambda_o10,estimator='RLS')
    imp.mseMap_toCSV(para_err_RLS_o10,'mse_RLS_o10.csv')
    mse_RLS_o10 = imp.experiment(polyx,polyy,sampx,sampy,paradict=opt_para_RLS_o10,method='RLS',plot_title=NAME_MAP['RLS']+' order 10')

    para_err_LASSO_o10,opt_para_LASSO_o10 = imp.model_selection(polyx,polyy,sampx,sampy,para_Lambda_o10,estimator='LASSO')
    imp.mseMap_toCSV(para_err_LASSO_o10,'mse_LASSO_o10.csv')
    mse_LASSO_o10 = imp.experiment(polyx,polyy,sampx,sampy,paradict=opt_para_LASSO_o10,method='LASSO',plot_title=NAME_MAP['LASSO']+' order 10')

    para_err_BR_o10,opt_para_BR_o10 = imp.model_selection(polyx,polyy,sampx,sampy,para_BR_o10,estimator='BR')
    imp.mseMap_toCSV(para_err_BR_o10,'mse_BR_o10.csv')
    mse_BR_o10 = imp.experiment(polyx,polyy,sampx,sampy,paradict=opt_para_BR_o10,method='BR',plot_title=NAME_MAP['BR']+' order 10')

if __name__ == "__main__":
    main()