#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 16:18:03 2017

@author: rakshitanagalla
"""

import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import calendar

def add_months(sourcedate,months):
    month = sourcedate.month - 1 + months
    year = int(sourcedate.year + month / 12 )
    month = month % 12 + 1
    day = min(sourcedate.day,calendar.monthrange(year,month)[1])
    return dt.date(year,month,day)


data = pd.read_csv('Recent_toBeUsed.csv')
data.drop(['e','YearMonth','Unnamed: 4', 'y', 'y_US','m_in_billion_rupees', 'm_US_in billion rupees','m_in_billion_rupees.1','i', 'i_US','m_US_in billion', 'Dummy', 'Dummy2'],inplace=True,axis=1)
DTTMFormat = '%d-%m-%Y'
data['DTTM'] = pd.to_datetime(data['DTTM'],format=DTTMFormat)
data.set_index('DTTM', inplace = 'True')
#data_diff = data.diff().dropna() #differencing


ForecastTime = dt.datetime.strptime('1-7-2006', DTTMFormat)


lag = 1
constant =1
M = 4 #Number of endogenous variables

outputPredDF = pd.DataFrame()
se = pd.DataFrame()
cnt = 0
while cnt < 110:
    train_period = [ForecastTime - relativedelta(years=10), add_months(ForecastTime,-1)]
    if cnt == 0:
        train = data[add_months(train_period[0],1):train_period[1]]
    else:
        train = data[train_period[0]:train_period[1]]
    test = data[ForecastTime:add_months(ForecastTime+relativedelta(years=1),-1)]
    
    X1 = np.matrix(pd.concat([train, train.shift(1)], axis=1).dropna()[:-1])
    X = np.c_[np.ones(len(X1)), X1 ]  
    Y = np.matrix(train[lag:])
    
    # OLS coefficients
    A_OLS = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y)) # This is the matrix of regression coefficients
    a_OLS = A_OLS.T.ravel().T        # This is the vector of coefficients, i.e. it holds
                              # that a_OLS = vec(A_OLS)
    SSE = np.dot( (Y - np.dot(X,A_OLS)).T , (Y - np.dot(X,A_OLS)) )
    K,M = A_OLS.shape
    T = X.shape[0]
    SIGMA_OLS = SSE/(T-K)
                    
    '''
    Prior Hyperparameters for Minesotta prior
    '''
    A_prior = np.zeros((K,M))   
    a_prior = np.matrix(A_prior.ravel()).T
        
    # Hyperparameters on the Minnesota variance of alpha
    a_bar_1 = 0.5;
    a_bar_2 = 0.5;
    a_bar_3 = 10**2;
        
    # Now get residual variances of univariate p-lag autoregressions. Here
    # we just run the AR(p) model on each equation,(if they have been specified for the original
    # VAR model)
    
    sigma_sq = np.zeros((M,1)); # vector to store residual variances
    for i in range(M):
        # Create lags of dependent variable in i-th equation
        #Ylag_i = X[:,[i+1,i+1+M]]
        Ylag_i = X[:,np.arange(constant+i,K,M)] 
        # Dependent variable in i-th equation
        Y_i = Y[:,i]
        # OLS estimates of i-th equation
        alpha_i = np.dot(np.linalg.inv(np.dot(Ylag_i.T,Ylag_i)),np.dot(Ylag_i.T,Y_i))
        sigma_sq[i,0] = np.dot((Y_i - np.dot(Ylag_i,alpha_i)).T,(Y_i - np.dot(Ylag_i,alpha_i)))/(T-lag+1)
    
    # Now define prior hyperparameters.
    # Create an array of dimensions K x M, which will contain the K diagonal
    # elements of the covariance matrix, in each of the M equations.
    V_i = np.zeros((K,M));
    
    # index in each equation which are the own lags
    ind = np.zeros((M,lag));
    for i in range(M):
        ind[i,:] = np.arange(constant+i,K,M)
    for i in range(M):  # for each i-th equation
        for j in range(K):   # for each j-th RHS variable
            if j==0:
                V_i[j,i] = a_bar_3*sigma_sq[i,0] # variance on constant                
            elif j in ind[i,:]:
                V_i[j,i] = a_bar_1/(np.ceil((j)/M)**2); # variance on own lags           
            else:
                for kj in range(M):
                    if j in ind[kj,:]: 
                        ll = kj                   
                V_i[j,i] = (a_bar_2*sigma_sq[i,0])/((np.ceil((j)/M)**2)*sigma_sq[ll,0]);           
    
    # Now V is a diagonal matrix with diagonal elements the V_i
    V_prior = np.diag(V_i.ravel());  # this is the prior variance of the vector a  
    
    # SIGMA is equal to the OLS quantity
    SIGMA = SIGMA_OLS;
    
    '''
    POSTERIORS 
    '''   
    #--------- Posterior hyperparameters of ALPHA and SIGMA with Minnesota Prior      
    V_post = np.linalg.inv(np.linalg.inv(V_prior) + np.kron(np.linalg.inv(SIGMA),np.dot(X.T,X)) );
    a_post = np.dot(V_post, np.dot(np.linalg.inv(V_prior),a_prior) + np.dot(np.kron(np.linalg.inv(SIGMA),np.dot(X.T,X)),a_OLS) )
    A_post = np.reshape(a_post, (K, M),order='F')
    
    '''
    PREDICTIVE INFERENCE 
    '''
    forecasts = np.zeros(12)
    test_new = train[-lag:]
    for step in range(12):
        X2 = np.matrix(pd.concat([test_new[-lag:], test_new[-lag:].shift(1)], axis=1).dropna())
        X_test = np.c_[np.ones(len(X2)), X2]
        forecasts[step] = np.dot(X_test,A_post)[0,0]
        test_new.loc[step+2] = np.array(np.dot(X_test,A_post))[0]
        
    final_pred = np.exp(forecasts)
    #final_pred = np.exp(np.r_[data['ln_e'][train_period[1]], forecasts].cumsum())[1:]

    
    #save predictions and errors 
    columName = str(ForecastTime.year)+'M'+str(ForecastTime.month)
    outputPredDF[columName+'_pred'] = final_pred
    se[columName+'_err'] = np.array((np.exp(test.ln_e) - final_pred)**2)
      
    cnt +=1
    ForecastTime = add_months(ForecastTime,1)
    print(train_period)

rmse = np.sqrt(se.mean(axis = 1)) #T_0 = 110











