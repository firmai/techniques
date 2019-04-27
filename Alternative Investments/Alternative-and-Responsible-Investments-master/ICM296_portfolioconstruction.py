
# # Preamp

# In[10]:

import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format

import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-whitegrid')
#print(plt.style.available)
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import matplotlib.mlab as mlab

import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter
#import pandas_datareader.data as web
#import datetime as dt
import statsmodels.tools

from scipy import stats
import time


# # Importing data

# In[11]:

#import data_to_pickle
start = time.time()

# Read pickles
import read_pickle as pick
import functions as func
# # Organisation of data
# Here, I organise all the stock prices into the possible investment universes. 
# This is done by constructing a dictionary
universe = {}
for i in range(0,len(pick.investmentuniverse.columns)):
    universe [str(i)] = pick.investmentuniverse[pick.investmentuniverse.columns[i]].dropna() 
    # Dropna to filter out any missing values


Portfolios = {}
years = []
for i in range (2002, 2016+1):
    years.append(i)
for i in range (2, 17):
    Portfolios [str(i)] = pick.icm[universe[str(i-2)]].loc[str(years[i-2])][1:]

# # Working with dividends
dividends_quarterly = pick.dividends.resample('Q').sum()


# # Portfolios

# ## Portfolio 2002

# In[12]:

test = dividends_quarterly['2001'][Portfolios['2'].keys()].iloc[2:]
summ = {}
for col in test:
    valid_col = test.dropna(axis = 1) #Finner alle kolonner som har 2 verdier
    valid_col = valid_col.iloc[-1] #Velger den siste verdien av disse
    summ [str(col)] = test[str(col)].sum() #Legger til alle kolonner med tilhørende sum i en dictionary
    summ = pd.DataFrame(summ, index = [0]) #Lager en ny Dataframe

summ [valid_col.index] = valid_col.values
summ = summ / pick.icm[Portfolios['2'].keys()]['2001'].iloc[-1] * 100
summ = summ.transpose()
summ = summ.sort_values(summ.columns[0], axis = 0,ascending = False)

list2 = summ[1:11]
list2.to_clipboard()
list2 = list2.index
print(list2)

P02 = pick.totret[list2]['2002']

P02.head()

P02.mean(axis = 1).to_clipboard()
P02.mean(axis = 1)


# ## Portfolio 2003

# In[13]:

test = dividends_quarterly['2002'][Portfolios['3'].keys()].iloc[2:]
summ = {}
for col in test:
    valid_col = test.dropna(axis = 1) #Finner alle kolonner som har 2 verdier
    valid_col = valid_col.iloc[-1] #Velger den siste verdien av disse
    summ [str(col)] = test[str(col)].sum() #Legger til alle kolonner med tilhørende sum i en dictionary
    summ = pd.DataFrame(summ, index = [0]) #Lager en ny Dataframe

summ [valid_col.index] = valid_col.values
summ = summ / pick.icm[Portfolios['3'].keys()]['2002'].iloc[-1] * 100
summ = summ.transpose()
summ = summ.sort_values(summ.columns[0], axis = 0,ascending = False)

list3 = summ[1:11]
list3.to_clipboard()
list3 = list3.index

P03 = pick.totret.loc['2003',list3]
P03.head()
P03.mean(axis = 1).to_clipboard()
P03.mean(axis = 1)


# ## Portfolio 2004

# In[14]:

test = dividends_quarterly['2003'][Portfolios['4'].keys()].iloc[2:]
summ = {}
for col in test:
    valid_col = test.dropna(axis = 1) #Finner alle kolonner som har 2 verdier
    valid_col = valid_col.iloc[-1] #Velger den siste verdien av disse
    summ [str(col)] = test[str(col)].sum() #Legger til alle kolonner med tilhørende sum i en dictionary
    summ = pd.DataFrame(summ, index = [0]) #Lager en ny Dataframe

summ [valid_col.index] = valid_col.values
summ = summ / pick.icm[Portfolios['4'].keys()]['2003'].iloc[-1] * 100
summ = summ.transpose()
summ = summ.sort_values(summ.columns[0], axis = 0,ascending = False)

list4 = summ[1:11]
list4.to_clipboard()
list4 = list4.index

P04 = pick.totret[list4]['2004']
P04.head(2)

P04.mean(1).to_clipboard()
P04.mean(1)


# ## Portfolio 2005

# In[15]:

test = dividends_quarterly['2004'][Portfolios['5'].keys()].iloc[2:]
summ = {}
for col in test:
    valid_col = test.dropna(axis = 1) #Finner alle kolonner som har 2 verdier
    valid_col = valid_col.iloc[-1] #Velger den siste verdien av disse
    summ [str(col)] = test[str(col)].sum() #Legger til alle kolonner med tilhørende sum i en dictionary
    summ = pd.DataFrame(summ, index = [0]) #Lager en ny Dataframe

summ [valid_col.index] = valid_col.values
summ = summ / pick.icm[Portfolios['5'].keys()]['2004'].iloc[-1] * 100
summ = summ.transpose()
summ = summ.sort_values(summ.columns[0], axis = 0,ascending = False)

list5 = summ[1:11]
list5.to_clipboard()
list5 = list5.index

P05 = pick.totret[list5]['2005']

P05.head(2)

P05.mean(1).to_clipboard()
P05.mean(1)


# ## Portfolio 2006

# In[16]:

test = dividends_quarterly['2005'][Portfolios['6'].keys()].iloc[2:]
summ = {}
for col in test:
    valid_col = test.dropna(axis = 1) #Finner alle kolonner som har 2 verdier
    valid_col = valid_col.iloc[-1] #Velger den siste verdien av disse
    summ [str(col)] = test[str(col)].sum() #Legger til alle kolonner med tilhørende sum i en dictionary
    summ = pd.DataFrame(summ, index = [0]) #Lager en ny Dataframe

summ [valid_col.index] = valid_col.values
summ = summ / pick.icm[Portfolios['6'].keys()]['2005'].iloc[-1] * 100
summ = summ.transpose()
summ = summ.sort_values(summ.columns[0], axis = 0,ascending = False)

list6 = summ[1:11]
list6.to_clipboard()
list6 = list6.index

P06 = pick.totret[list6]['2006']
P06 ['UKX Index'] = pick.icm['UKX Index']['2006'].resample('m').last().pct_change(1)+1
P06['UKX Index'].iloc[:7] = np.nan

P06.head(8)
P06.mean(1).to_clipboard()
P06.mean(1)


# ## Portfolio 2007

# In[17]:

test = dividends_quarterly['2006'][Portfolios['7'].keys()].iloc[2:]
summ = {}
for col in test:
    valid_col = test.dropna(axis = 1) #Finner alle kolonner som har 2 verdier
    valid_col = valid_col.iloc[-1] #Velger den siste verdien av disse
    summ [str(col)] = test[str(col)].sum() #Legger til alle kolonner med tilhørende sum i en dictionary
    summ = pd.DataFrame(summ, index = [0]) #Lager en ny Dataframe

summ [valid_col.index] = valid_col.values
summ = summ / pick.icm[Portfolios['7'].keys()]['2006'].iloc[-1] * 100
summ = summ.transpose()
summ = summ.sort_values(summ.columns[0], axis = 0,ascending = False)

list7 = summ[:10]
list7.to_clipboard()
list7 = list7.index

P07 = pick.totret[list7]['2007']
P07 ['UKX Index'] = pick.icm['UKX Index']['2007'].resample('m').last().pct_change(1)+1
P07['UKX Index'][:'2007-06-30'] = np.nan # Alliance was acquired by a P/E company

P07.mean(1).to_clipboard()
P07.mean(1)


# ## Portfolio 2008

# In[18]:

test = dividends_quarterly['2007'][Portfolios['8'].keys()].iloc[2:]
summ = {}
for col in test:
    valid_col = test.dropna(axis = 1) #Finner alle kolonner som har 2 verdier
    valid_col = valid_col.iloc[-1] #Velger den siste verdien av disse
    summ [str(col)] = test[str(col)].sum() #Legger til alle kolonner med tilhørende sum i en dictionary
    summ = pd.DataFrame(summ, index = [0]) #Lager en ny Dataframe

summ [valid_col.index] = valid_col.values
summ = summ / pick.icm[Portfolios['8'].keys()]['2007'].iloc[-1] * 100
summ = summ.transpose()
summ = summ.sort_values(summ.columns[0], axis = 0,ascending = False)

list8 = summ[:10]
list8.to_clipboard()
list8 = list8.index

P08 = pick.totret[list8]['2008']
P08 ['UKX Index'] = pick.icm['UKX Index']['2008'].resample('m').last().pct_change(1)+1
P08 ['UKX Index'][:'2008-10-31'] = np.nan # Alliance was acquired by a P/E company

P08.mean(1).to_clipboard()
P08.mean(1)


# ## Portfolio 2009

# In[19]:

test = dividends_quarterly['2008'][Portfolios['9'].keys()].iloc[2:]
summ = {}
for col in test:
    valid_col = test.dropna(axis = 1) #Finner alle kolonner som har 2 verdier
    valid_col = valid_col.iloc[-1] #Velger den siste verdien av disse
    summ [str(col)] = test[str(col)].sum() #Legger til alle kolonner med tilhørende sum i en dictionary
    summ = pd.DataFrame(summ, index = [0]) #Lager en ny Dataframe

summ [valid_col.index] = valid_col.values
summ = summ / pick.icm[Portfolios['9'].keys()]['2008'].iloc[-1] * 100
summ = summ.transpose()
summ = summ.sort_values(summ.columns[0], axis = 0,ascending = False)

list9 = summ[:10]
list9.to_clipboard()
list9 = list9.index

P09 = pick.totret[list9]['2009']
P09.head(4)
P09.mean(1).to_clipboard()


# ## Portfolio 2010

# In[20]:

test = dividends_quarterly['2009'][Portfolios['10'].keys()].iloc[2:]
summ = {}
for col in test:
    valid_col = test.dropna(axis = 1) #Finner alle kolonner som har 2 verdier
    valid_col = valid_col.iloc[-1] #Velger den siste verdien av disse
    summ [str(col)] = test[str(col)].sum() #Legger til alle kolonner med tilhørende sum i en dictionary
    summ = pd.DataFrame(summ, index = [0]) #Lager en ny Dataframe

summ [valid_col.index] = valid_col.values
summ = summ / pick.icm[Portfolios['10'].keys()]['2009'].iloc[-1] * 100
summ = summ.transpose()
summ = summ.sort_values(summ.columns[0], axis = 0,ascending = False)

list10 = summ[:10]
list10.to_clipboard()
list10 = list10.index

P10 = pick.totret[list10]['2010']
P10.mean(1).to_clipboard()


# ## Portfolio 2011

# In[21]:

test = dividends_quarterly['2010'][Portfolios['11'].keys()].iloc[2:]
summ = {}
for col in test:
    valid_col = test.dropna(axis = 1) #Finner alle kolonner som har 2 verdier
    valid_col = valid_col.iloc[-1] #Velger den siste verdien av disse
    summ [str(col)] = test[str(col)].sum() #Legger til alle kolonner med tilhørende sum i en dictionary
    summ = pd.DataFrame(summ, index = [0]) #Lager en ny Dataframe

summ [valid_col.index] = valid_col.values
summ = summ / pick.icm[Portfolios['11'].keys()]['2010'].iloc[-1] * 100
summ = summ.transpose()
summ = summ.sort_values(summ.columns[0], axis = 0,ascending = False)

list11 = summ[:10]
list11.to_clipboard()
list11 = list11.index

P11 = pick.totret[list11]['2011']
P11.mean(1).to_clipboard()


# ## Portfolio 2012

# In[22]:

test = dividends_quarterly['2011'][Portfolios['12'].keys()].iloc[2:]
summ = {}
for col in test:
    valid_col = test.dropna(axis = 1) #Finner alle kolonner som har 2 verdier
    valid_col = valid_col.iloc[-1] #Velger den siste verdien av disse
    summ [str(col)] = test[str(col)].sum() #Legger til alle kolonner med tilhørende sum i en dictionary
    summ = pd.DataFrame(summ, index = [0]) #Lager en ny Dataframe

summ [valid_col.index] = valid_col.values
summ = summ / pick.icm[Portfolios['12'].keys()]['2011'].iloc[-1] * 100
summ = summ.transpose()
summ = summ.sort_values(summ.columns[0], axis = 0,ascending = False)

list12 = summ[:10]
list12.to_clipboard()
list12 = list12.index

P12 = pick.totret[list12]['2012']
P12.mean(1).to_clipboard()
P12.to_clipboard()


# ## Portfolio 2013

# In[23]:

test = dividends_quarterly['2012'][Portfolios['13'].keys()].iloc[2:]
summ = {}
for col in test:
    valid_col = test.dropna(axis = 1) #Finner alle kolonner som har 2 verdier
    valid_col = valid_col.iloc[-1] #Velger den siste verdien av disse
    summ [str(col)] = test[str(col)].sum() #Legger til alle kolonner med tilhørende sum i en dictionary
    summ = pd.DataFrame(summ, index = [0]) #Lager en ny Dataframe

summ [valid_col.index] = valid_col.values
summ = summ / pick.icm[Portfolios['13'].keys()]['2012'].iloc[-1] * 100
summ = summ.transpose()
summ = summ.sort_values(summ.columns[0], axis = 0,ascending = False)

list13 = summ[:10]
list13.to_clipboard()
list13 = list13.index

P13 = pick.totret[list13]['2013']
P13.mean(1).to_clipboard()


# ## Portfolio 2014

# In[24]:

test = dividends_quarterly['2013'][Portfolios['14'].keys()].iloc[2:]
summ = {}
for col in test:
    valid_col = test.dropna(axis = 1) #Finner alle kolonner som har 2 verdier
    valid_col = valid_col.iloc[-1] #Velger den siste verdien av disse
    summ [str(col)] = test[str(col)].sum() #Legger til alle kolonner med tilhørende sum i en dictionary
    summ = pd.DataFrame(summ, index = [0]) #Lager en ny Dataframe

summ [valid_col.index] = valid_col.values
summ = summ / pick.icm[Portfolios['14'].keys()]['2013'].iloc[-1] * 100
summ = summ.transpose()
summ = summ.sort_values(summ.columns[0], axis = 0,ascending = False)

list14 = summ[:10]
list14.to_clipboard()
list14 = list14.index

P14 = pick.totret[list14]['2014']
P14.mean(1).to_clipboard()


# ## Portfolio 2015

# In[25]:

test = dividends_quarterly['2014'][Portfolios['15'].keys()].iloc[2:]
summ = {}
for col in test:
    valid_col = test.dropna(axis = 1) #Finner alle kolonner som har 2 verdier
    valid_col = valid_col.iloc[-1] #Velger den siste verdien av disse
    summ [str(col)] = test[str(col)].sum() #Legger til alle kolonner med tilhørende sum i en dictionary
    summ = pd.DataFrame(summ, index = [0]) #Lager en ny Dataframe

summ [valid_col.index] = valid_col.values
summ = summ / pick.icm[Portfolios['15'].keys()]['2014'].iloc[-1] * 100
summ = summ.transpose()
summ = summ.sort_values(summ.columns[0], axis = 0,ascending = False)

list15 = summ[:10]
list15.to_clipboard()
list15 = list15.index

P15 = pick.totret[list15]['2015']
P15.mean(1).to_clipboard()


# ## Portfolio 2016

# In[26]:

test = dividends_quarterly['2015'][Portfolios['16'].keys()].iloc[2:]
summ = {}
for col in test:
    valid_col = test.dropna(axis = 1) #Finner alle kolonner som har 2 verdier
    valid_col = valid_col.iloc[-1] #Velger den siste verdien av disse
    summ [str(col)] = test[str(col)].sum() #Legger til alle kolonner med tilhørende sum i en dictionary
    summ = pd.DataFrame(summ, index = [0]) #Lager en ny Dataframe

summ [valid_col.index] = valid_col.values
summ = summ / pick.icm[Portfolios['16'].keys()]['2015'].iloc[-1] * 100
summ = summ.transpose()
summ = summ.sort_values(summ.columns[0], axis = 0,ascending = False)

list16 = summ[:10]
list16.to_clipboard()
list16 = list16.index

P16 = pick.totret[list16]['2016']
P16.mean(1).to_clipboard()

