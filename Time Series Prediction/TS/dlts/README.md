# Deep Learning Architecture for time series forecasting

The goal of this project is to understand how deep learning architecture like Long Short Term Memory networks can be leveraged to improve the forecast of multivariate econometric time series. In addition to compring LSTM's performance to traditional time series models like ARIMA and VAR, bayesian approaches are also explored.  

A summary of the results can be found in [this presentation.](Final_presentation.pdf)

## Description of contents:  

stationarity_acf.py: Test the stationarity of time series using Dicky-fuller test and autocorrelation and partial autocorrelation plots  
arima.py: Implementation of ARIMA model to compute multi-step rolling forecasts  
VAR.py: Implementation of ARIMA model to compute multi-step rolling forecasts  
bvar.py: Implementation of Bayesian Vector Autogressive model to compute multi-step rolling forecasts  
bvar_rolling.py: Implementation of Bayesian Vector Autogressive model to compute multi-step rolling forecasts  
LSTM_rolling.py: Implementation of LSTM to compute multi-step rolling forecasts  
Recent_toBeUsed.csv: Data file 
