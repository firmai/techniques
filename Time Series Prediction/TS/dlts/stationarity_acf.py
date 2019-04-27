#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:18:36 2017

@author: rakshitanagalla
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('Recent_toBeUsed.csv')
data.drop(['YearMonth','Unnamed: 4', 'y', 'y_US','m_in_billion_rupees', 'm_US_in billion rupees','m_in_billion_rupees.1','i', 'i_US','m_US_in billion', 'Dummy', 'Dummy2'],inplace=True,axis=1)
DTTMFormat = '%m-%d-%Y'
data['DTTM'] = pd.to_datetime(data['DTTM'],format=DTTMFormat)

#Test stationarity
from statsmodels.tsa.stattools import adfuller

print ('Results of Dickey-Fuller Test:')
dftest = adfuller(data['ln_m_diff'].diff()[1:], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput.values)

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(data['ln_e'].diff()[1:], nlags=20)
lag_pacf = pacf(data['ln_e'].diff()[1:], nlags=20, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data['ln_e'].diff()[1:])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data['ln_e'].diff()[1:])),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data['ln_e'].diff()[1:])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data['ln_e'].diff()[1:])),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

#Model

data.set_index('DTTM', inplace = 'True')

train = data['e'][:161]
test = data['e'][161:]

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(train, order=(2, 1, 2))  #(p,d,q)
model_fit = model.fit(disp=0) 

predicted = model_fit.forecast(steps = 80)[0]
#plt.plot(data['ln_e'].diff()[1:])
#plt.plot(results_ARIMA.fittedvalues, color='red')
#plt.title('RMSE: %.4f'% sum((results_ARIMA.fittedvalues-data['ln_e'].diff()[1:])**2))

#print('RMSE: %.4f'% sum((results_ARIMA.fittedvalues-data['ln_e'].diff()[1:])**2))
print('RMSE: %.4f'% ((predicted - test) ** 2).mean())
