# -*- coding: utf-8 -*-
"""
Created on Mon May 01 01:49:07 2017

@author: Rakshita
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import calendar
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from scipy.ndimage.interpolation import shift

def add_months(sourcedate,months):
    month = sourcedate.month - 1 + months
    year = int(sourcedate.year + month / 12 )
    month = month % 12 + 1
    day = min(sourcedate.day,calendar.monthrange(year,month)[1])
    return dt.date(year,month,day)

# convert an array of values into a dataset matrix
def create_dataset(a, look_back):
    dataX, dataY = [], []
    dataX = a[:-look_back]
    for l in range(look_back-1):
        l+=1
        dataX = np.c_[dataX,shift(a, -l)[:-look_back]]
    dataY = shift(a, -look_back)[:-look_back]
    dataY = np.reshape(dataY, (len(dataY),1))
    return np.array(dataX), np.array(dataY)

data = pd.read_csv('Recent_toBeUsed.csv')
data.drop(['ln_e','YearMonth','Unnamed: 4', 'y', 'y_US','m_in_billion_rupees', 'm_US_in billion rupees','m_in_billion_rupees.1','i', 'i_US','m_US_in billion', 'Dummy', 'Dummy2'],inplace=True,axis=1)
DTTMFormat = '%d-%m-%Y'
data['DTTM'] = pd.to_datetime(data['DTTM'],format=DTTMFormat)
data.set_index('DTTM', inplace = 'True')

ForecastTime = dt.datetime.strptime('1-7-2006', DTTMFormat)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaler3 = MinMaxScaler(feature_range=(0, 1))

'''
Change as required
'''

n_hidden = 1
look_back = 13

cnt =0
outputPredDF = pd.DataFrame()
se = pd.DataFrame()
while cnt < 110:
    train_period = [ForecastTime - relativedelta(years=10), add_months(ForecastTime,-1)]
    if cnt == 0:
        Train = data[add_months(train_period[0],1):train_period[1]]
    else:
        Train = data[train_period[0]:train_period[1]]
    Test = data[ForecastTime:add_months(ForecastTime+relativedelta(years=1),-1)]
    
    N_train = len(Train)
    
    train =  scaler.fit_transform(Train['e']) 
    trainX, trainY = create_dataset(train, look_back)

    train1 =  scaler1.fit_transform(Train['ln_y_diff'])
    trainX1, trainY1 = create_dataset(train1, look_back)
    
    train2 =  scaler2.fit_transform(Train['ln_m_diff'])
    trainX2, trainY2 = create_dataset(train2, look_back)
   
    train3 =  scaler3.fit_transform(Train['i-i_US'])
    trainX3, trainY3 = create_dataset(train3, look_back)
    
    N_train_new = trainX.shape[0]
    
    trainX_i=np.zeros(shape=[N_train_new,look_back,4])
    trainX_i[:,:,0] = trainX
    trainX_i[:,:,1] = trainX1
    trainX_i[:,:,2] = trainX2
    trainX_i[:,:,3] = trainX3
    
    trainY_i=np.zeros(shape=[N_train_new,4])
    trainY_i[:,0]=trainY[:,0]
    trainY_i[:,1]=trainY1[:,0]
    trainY_i[:,2]=trainY2[:,0]
    trainY_i[:,3]=trainY3[:,0]

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(n_hidden, input_shape=(look_back,4)))
    model.add(Dense(4))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX_i, trainY_i, epochs=60, batch_size=1, verbose=2)# epoch = 60
    
    # make predictions
    forecast = np.zeros(12)
    test_new = trainX_i[-1,:,:]
    for step in range(12):
        X_test = np.reshape(test_new[-look_back:],(1,look_back,4))
        forecasts = model.predict(X_test)
        forecast[step] = forecasts[0,0]
        test_new = np.r_[test_new, forecasts]
    
    final_forecast = scaler.inverse_transform(forecast)

    #save predictions and errors 
    columName = str(ForecastTime.year)+'M'+str(ForecastTime.month)
    outputPredDF[columName+'_pred'] = final_forecast
    se[columName+'_err'] = np.array((Test['e'] - final_forecast)**2)
      
    cnt +=1
    ForecastTime = add_months(ForecastTime,1)
    #print(train_period)
    print (cnt)
    
rmse = np.sqrt(se.mean(axis = 1)) #T_0 = 110
    
      
    
