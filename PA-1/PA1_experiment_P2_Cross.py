import PA1_Imp as imp
import numpy as np
import pandas as pd
import math as m
import os
import matplotlib.pyplot as plt


# cd D:\\OneDrive\\文档\\cityu\\MachineLearning\\MLAssignment\\
# cd E:\\OneDrive\\文档\\cityu\\MachineLearning\\MLAssignment\\

NAME_MAP = imp.NAME_MAP

def main():
    testx,testy,trainx,trainy = imp.load_dataset_P2()

    nopara_dict={'function':'id','order':1,'Lambda':0}
    lambda_candict = {'Lambda':[0.1,0.25,0.5,1,2,5],'function':['id'],'order':[1]}
    BR_candict = {'alpha':[0.1,0.5,1,5],'sigma':[0.1,0.5,1,5],'function':['id'],'order':[1]}

    opt_params = {}
    opt_params['LS'] = nopara_dict
    opt_params['RR'] = nopara_dict
        
    predict_dict={}        
    errors = {}

    # model selection

    plt.plot(testy, 'ko',label='True Values')
    plt.legend()


    for key, value in NAME_MAP.items():

        if key=='RR': continue
        #print (key)
        if (key=='RLS' or key =='LASSO'):
            para_err,opt_para = imp.model_selection(testx,testy,trainx,trainy,lambda_candict,estimator=key)
            imp.mseMap_toCSV(para_err,'P2_mse_normal'+key+'.csv')
            opt_params[key] = opt_para      
        elif(key == 'BR'):
            para_err,opt_para = imp.model_selection(testx,testy,trainx,trainy,BR_candict,estimator=key)
            imp.mseMap_toCSV(para_err,'P2_mse_normal'+key+'.csv')
            opt_params[key] = opt_para 
        
        params = opt_params[key]
        mse,prediction = imp.experiment(testx,testy,trainx,trainy,paradict=params,method=key,plot_title='P2_cross'+value,show_plot=False)
        predict_dict[key] = prediction        
        mae = imp.mae(testy,prediction)    
        #imp.mseMap_toCSV({'mse':mse,'mae':mae},'P2_report'+key+'.csv')
        errors[key]=[mse,mae]

        
        plt.plot(prediction,'o',label=key)
        plt.legend()


        #imp.learning_curve(testx,testy,trainx,trainy,paradict=params,subset=[0.2,0.4,0.6,0.8,1],repeat=10,method=key,plot_title='Learning Curve P2 '+value,show_plot=False)
    epd=pd.DataFrame(errors)
    epd.to_csv(os.path.join('PA-1','plots','err_cross.csv'))
    plt.savefig(os.path.join('PA-1','plots','cross_fun'+'.jpg'))
    
if __name__ == "__main__":
    main()