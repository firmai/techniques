# In[1]:

get_ipython().magic('matplotlib inline')
#%% Preamp
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" # For reference, the options for that are 'all', 'none', 'last' and 'last_expr'

import seaborn as sns
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.5f}'.format

import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-whitegrid')
#print(plt.style.available)
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import matplotlib.mlab as mlab
from matplotlib.ticker import FuncFormatter

import statsmodels.api as sm

import statsmodels.tools

from scipy import stats
import time


# # Importing data

# In[2]:

start = time.time()
#import data_to_pickle # Used to import data and read into pickles
# Read pickles
import read_pickle as pick
import functions as func

# Organisation of data
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
    
# Dividends
dividends_quarterly = pick.dividends.resample('Q').sum()

# Python file with construction of portfolios
import ICM296_portfolioconstruction as pc # pc = portfolio_constructor

# Risk-free rate
rf = pd.read_pickle('rf')
rf = rf.iloc[:509]
rf = rf / 100
rf['IUMAJNB'] = rf['IUMAJNB'].map(lambda x: np.log((1+ x * (91/365.25))**(30.4375/91) ))
rf = rf * 100
rf.columns = rf.columns.map(lambda x: x + str(' (%)'))

rf.head()
rf.plot()


# # Regressions

# ## Data Prep

# In[3]:

Benchmark = pick.icm['UKX Index'].copy()
Benchmark = Benchmark.fillna(method = 'bfill')
Benchmark = Benchmark.pct_change(1)+1
Benchmark = Benchmark.iloc[1:]
Benchmark = np.log(Benchmark)*100
Benchmark = Benchmark.resample('M').sum()
Benchmark = Benchmark['2002':'2016']

df = pd.read_pickle('df')
df['Dates'] = pd.DatetimeIndex(freq = 'M', start='2002', end = '2017')
df.set_index('Dates', inplace = True)
df = np.log(df)*100
df['FTSE 100'] = Benchmark
df.columns = ['DoD', 'FTSE 100']
df ['DoD - FTSE 100'] = df['DoD'] - df ['FTSE 100']
df[['PDoD', 'PFTSE 100']]  =  np.exp(df[['DoD','FTSE 100']]/100)
df['PPDoD'] = func.valuecalculator(df['PDoD'])[1:]
Benchmark_value = func.valuecalculator(np.exp(Benchmark/100))[1:]
df['PPFTSE 100'] = Benchmark_value
df.columns = ['DoD (%)', 'FTSE 100 (%)', 
              'DoD - FTSE 100 (%)', 'PDoD', 
              'PFTSE 100', 'PPDoD', 'PPFTSE 100']
df ['Risk-Free Rate (%)'] = rf.loc['2001-12-30':'2016',:].shift(1).iloc[1:] 
# Risk-free rate is shifted one month forward to reflect that the rate is for the next month 

df.tail()


# ## Data prep

# In[4]:

fama = pd.read_pickle('fama')
fama.set_index(pd.DatetimeIndex(freq = 'M', start='1980-10', end = '2016-7'), inplace = True)
fama = fama.loc['2002':,:]
fama ['DoD-Rf'] = (df.loc[:'2016-6','DoD (%)'] / 100) - (df.loc[:'2016-6','Risk-Free Rate (%)'] / 100)
fama ['Rm-Rf'] = (df.loc[:'2016-6','FTSE 100 (%)'] / 100) - (df.loc[:'2016-6','Risk-Free Rate (%)'] / 100)
fama = fama.loc[:,['DoD-Rf','Rm-Rf','SMB','HML', 'UMD']]

fama.head()


# ## CAPM

# In[5]:

ydata = fama.loc[:,'DoD-Rf']
xdata = fama.loc[:,fama.columns[1]]
CAPM_results = sm.OLS(ydata, sm.add_constant(xdata)).fit()

print(CAPM_results.summary(yname = 'DoD Excess Returns')) 


# ### CAPM Plot

# In[6]:

plt.close('all')
sns.set(color_codes=True)
fama.to_clipboard()
ax = sns.regplot(xdata,ydata, scatter = True)
ax.set_facecolor('white')
ax.grid(False)
xaxis = ax.get_xlim()

ax.annotate('Regression line: y = {:.5f} +'
            '{:.3f}x\nR-Square: {:.4f}'.format(CAPM_results.params[0],
                                               CAPM_results.params[1],
                                               CAPM_results.rsquared),
            (xaxis[0],0.93*ax.get_ylim()[1]))

ax.set_ylim(top = ax.get_ylim()[1]*1.1)
plt.title('CAPM ', loc = 'left', fontweight = 'bold')

plt.axhline(y = 0, color = 'k', linewidth = 0.5)

savepath = 'C:/Users/Christopher/Dropbox/2. ICMA Centre/PROTEXT test/ICM_296/CAPM_plot.png'
plt.savefig(savepath,transparent = True, dpi = 300, bbox_inches="tight");


# ## Fama French

# In[7]:

ydata = fama.loc[:,'DoD-Rf']
xdata = fama.loc[:,fama.columns[1:-1]]

FF_results = sm.OLS(ydata, sm.add_constant(xdata)).fit()
print(FF_results.summary(yname = 'DoD Excess Returns'));


# ## Carhart

# In[8]:

ydata = fama.loc[:,'DoD-Rf']
xdata = fama.loc[:,fama.columns[1:]]

CAR_results = sm.OLS(ydata, sm.add_constant(xdata)).fit()
print(CAR_results.summary(yname = 'DoD Excess Returns'));


# # Risk Measures

# ## Test Statistics

# In[9]:

series = df.loc[:,'DoD (%)'] - df.loc[:,'Risk-Free Rate (%)']

d = series.mean()
s = series.std(ddof = 1)
n = np.sqrt(len((series)))

t = d/s*n

pval = stats.t.sf(np.abs(t), n**2-1)*2 # Must be two-sided as we're looking at <> 0

if pval <= 0.05:
    print('t-statistics = {:.2f}, P-value = {:.3f}'
          ' (Statistically significant)'.format(t, pval))
else:
    print('t-statistics = {:.2f}, P-value = {:.3f}'
          ' (Not statistically significant)'.format(t, pval))


# ## Maximum Drawdown

# In[10]:

# A drawdown is a measurement of decline from an assets peak value to its 
# lowest point over a period of time. The drawdown is usually expressed as a 
# percentage from top to bottom. It can be measured on any asset including 
# individual stocks or sectors. However, it is most valuable as a measurement 
# of portfolio risk.
max_drawdownDoD = func.DD_measure(df['PPDoD'])
max_drawdownBench = func.DD_measure(df['PPFTSE 100'])
print("Max DoD drawdown is {:.2f}%".format(max_drawdownDoD))
print("Max benchmark drawdown is {:.2f}%".format(max_drawdownBench))


# ## VaR and Expected Shortfall

# In[11]:

VV = func.VaR (df.loc[:,'DoD (%)']) # Self-made function
VV.head()
VV.loc[:,['DoD (%)','Cumulative Weight']].where(VV['Cumulative Weight'] <= .05).dropna()
VV2 = func.VaR (df.loc[:,'FTSE 100 (%)'])
VV2.loc[:,['FTSE 100 (%)','Cumulative Weight']].where(VV2['Cumulative Weight'] <= .05).dropna()

print("The expected shortfall is: ",VV.loc[:,['DoD (%)']].where(VV['Cumulative Weight'] < .05).dropna().mean())
print("The expected shortfall is: ",VV2.loc[:,['FTSE 100 (%)']].
      where(VV2['Cumulative Weight'] < .05).dropna().mean())


# ## Portfolio Turnover

# In[12]:

lenght = len(np.unique(pick.DoDportfolios.values))
print("\n \nThe number of constituting companies in the DoD is: {}".format(lenght))
DoD = pick.DoDportfolios.copy()

turnover = [0]
for i in range(2002,2016):
    temp = []
    for item in DoD[i]:

        if item in str(DoD[i+1]):
            temp.append(1)
        else:
            temp.append(0)
    turnover.append((10-sum(temp))/10)

turnover_mean = np.mean(turnover[1:])  * 100 
transactioncost = 2*turnover_mean*0.01

print('The average turnover in the period 2002 - 2016 '
      'is: {:.3f}%'.format(turnover_mean))
print('Therefore, turnover costs equates to: '
      '{:.2f}%'.format(transactioncost))


# ## Skewness and Kurtosis

# In[13]:

# The bias = False is due to different normalizations. 
# Scipy by default does not correct for bias
skew = stats.skew(fama.loc[:,'DoD-Rf'], bias = False)
skew2 = stats.skew(fama.loc[:,'Rm-Rf'], bias = False)

kurt = stats.kurtosis(fama.loc[:,'DoD-Rf'], bias = False)
kurt2 = stats.kurtosis(fama.loc[:,'Rm-Rf'], bias = False)

print('\n(DoD), (FTSE 100)\n----------------------------'
      '\nSkew: ({:.4f}), ({:.4f})'
      '\nKurtosis: ({:.4f}), ({:.4f})'
      .format(skew, skew2,
             kurt, kurt2))


# ### Jarque-Bera

# In[14]:

semi = (fama.loc[:,'DoD-Rf'])  # DoD portfolio excess returns
semi2 = fama.loc[:,'Rm-Rf'] # FTSE 100 excess returns

S = float(semi.shape[0]) / 6 * (skew**2 + 0.25*((kurt-3)**2)) # Test statistics
t = stats.chi2(2).ppf(0.95) # Threshold level
if S < t:
    print ("Not enough evidence to reject DoD as Normal "
           "according to the Jarque-Bera test. S = {:.4f} < {:.4f}".format(S,t))
else:
    print ("Reject that DoD is Normal according to "
           "the Jarque-Bera test; S = {:.4f} > {:.4f}".format(S,t))
    
S = float(semi2.shape[0]) / 6 * (skew2**2 + 0.25*((kurt2-3)**2)) # Test statistics
t = stats.chi2(2).ppf(0.95) # Threshold level
if S < t:
    print ("Not enough evidence to reject FTSE 100 as "
           "Normal according to the Jarque-Bera test. S = {:.4f} < {:.4f}".format(S,t))
else:
    print ("Reject that FTSE 100 is Normal according to the "
           "Jarque-Bera test; S = {:.4f} > {:.4f}".format(S,t))
    


# ## Corr, Variance, STD, SV and SSD

# ### Correlation

# In[15]:

fama.loc[:,fama.columns[:2]].corr()
fama.loc[:,fama.columns[:2]].cov()*12*100


# ### Variance

# In[16]:

var = np.var(semi, ddof = 1)
var2 = np.var(semi2, ddof = 1)
std = np.std(semi, ddof = 1) * np.sqrt(12) * 100
std2 = np.std(semi2, ddof = 1) * np.sqrt(12) * 100

print('\n(DoD), (FTSE 100)\n----------------------------'
      '\nVariance: ({:.4f}), ({:.4f})'
      '\nStandard Deviation: ({:.4f}), ({:.4f}) *Figures are annualised'
      .format(var, var2,
             std, std2))


# ### Semi-Variance and Semi-Standard Deviation

# In[17]:

# Based on the LPM using the average excess return as minimal acceptable return, the 
# concepts of semi-variance (SV) and semi-standard
# deviation (SSD) can be calculated

threshold1 = np.mean(semi)
threshold2 = np.mean(semi2)

semi_variance1 = func.LPM(semi,threshold1,2) 
# This is for the DoD porfolio, using the mean as the threshold
semi_variance2 = func.LPM(semi2,threshold2,2) 
# This is for the FTSE 100 porfolio, using the mean as the threshold

Semi_std_DoD, Semi_std_Bench = np.sqrt(semi_variance1), np.sqrt(semi_variance2)

print("Semi-variance is {:.3f}% for DoD, and {:.3f}% for FTSE 100".
      format(semi_variance1*100,semi_variance2*100))


# ## Relative Risk Measures 

# In[18]:

DoD_excess_mean = fama.loc[:,'DoD-Rf'].mean()
Benchmark_excess_mean = fama.loc[:,'Rm-Rf'].mean()
DoD_excess_std = fama.loc[:,'DoD-Rf'].std(ddof = 1)
Benchmark_excess_std = fama.loc[:,'Rm-Rf'].std(ddof = 1)


# ### Sharpe:

# In[19]:

sharpeDoD, sharpeBench = [DoD_excess_mean / DoD_excess_std,
                          Benchmark_excess_mean / Benchmark_excess_std]
sharpeDoD, sharpeBench


# ### RAPA

# In[20]:

RAPA = DoD_excess_mean * Benchmark_excess_std / DoD_excess_std
RAPA


# ### Treynor:

# In[21]:

TreynorDoD, TreynorBench = [DoD_excess_mean / CAR_results.params[1],
                            Benchmark_excess_mean / 1] 
# Beta from CARHART regression
TreynorDoD, TreynorBench


# ### Sortino:

# In[22]:

SortinoDoD, SortinoBench = [DoD_excess_mean / Semi_std_DoD, 
                            Benchmark_excess_mean / Semi_std_Bench]


# ### Probability of Shortfall and Return on Probability of Shortfall

# In[23]:

prob_of_shortfallDoD = func.LPM(semi, threshold1,0)
prob_of_shortfallBench = func.LPM(semi2, threshold2,0)
return_on_probability_shortfallDoD = DoD_excess_mean / prob_of_shortfallDoD
return_on_probability_shortfallBench = Benchmark_excess_mean / prob_of_shortfallBench


# ## Summary

# In[24]:

#%% Arithmetic returns
test = pd.DataFrame(df.loc[:,df.columns[0]]/100)


test ['RF'] = rf.loc['2002':'2016','IUMAJNB (%)']/100
test = test.loc['2002':'2016-06',:]
mean1 = test.loc[:,test.columns[0]].mean()*12 * 100

mean2 = (test.loc[:,test.columns[0]] - test.loc[:,test.columns[1]]).mean()*12 * 100
std2 = (test.loc[:,test.columns[0]] - test.loc[:,test.columns[1]]).std(ddof = 1)*np.sqrt(12) * 100

print('Arithmetic mean: {:.2}%'
      '\nArithmetic excess mean: {:.2}%'.format(mean1, mean2))

SterlingDoD = DoD_excess_mean * 11.6 / (np.abs(max_drawdownDoD)/100)
SterlingBench = Benchmark_excess_mean * 11.6 / (np.abs(max_drawdownBench)/100)
BurkeDoD = DoD_excess_mean * 11.6 / (np.sqrt((np.abs(max_drawdownDoD)/100)))
BurkeBench = Benchmark_excess_mean * 11.6 / (np.sqrt((np.abs(max_drawdownBench)/100)))
print('\n\n--Summary of risk measures--'
      '\n---(DoD, FTSE 100)---'
      '\nSharpe ratios: {:.4f}, {:.4f}'
      '\n RAPA: {:.4f}\nTreynor ratios: {:.4f}, {:.4f}\nSortino ratios: {:.4f}, {:.4f}'
      '\nProbability of Shortfall: {:.2f}%, {:.2f}%\nReturn on Prob. of Shortfall: {:.4f}, {:.4f}'
      '\nSterling ratios: {:.4}, {:.4}\nBurke ratios: {:.4}, {:.4}'
      '\nMaximum Drawdown ratios: {:.4}%, {:.4}%'
      '\n------------------------------------------'
      .format(sharpeDoD, sharpeBench, RAPA,  TreynorDoD,  TreynorBench,  
              SortinoDoD,  SortinoBench,  
              100*prob_of_shortfallDoD,  100*prob_of_shortfallBench, 
              return_on_probability_shortfallDoD, return_on_probability_shortfallBench, 
              SterlingDoD, SterlingBench, BurkeDoD, BurkeBench, 
              max_drawdownDoD, max_drawdownBench))


# # Plots

# ## Portfolio values

# In[25]:

plt.clf
plt.cla

df2 = df.loc['2002':'2016',['PPDoD','PPFTSE 100']].copy()
df2.plot(figsize=(10,5))
plt.title('Portfolio values', loc = 'left', fontweight = 'bold')
plt.ylabel('Value')

plt.xlabel('')

ax = plt.gca()
ax.legend(loc=2, fancybox=False, shadow=True, ncol=8)
ax.xaxis.grid(False)
ax.yaxis.grid(alpha = 0.8) # how visible is the lines

ax.axes.get_yaxis().set_major_formatter(
    FuncFormatter(lambda x, p: format(int(x), ',')))
ax.axes.set_facecolor('white')
# plt.figure(figsize=(20,10))

# Saving the figure
savepath = 'C:/Users/Christopher/Dropbox/2. ICMA Centre/PROTEXT test/ICM_296/Portfolio_values.png'
plt.savefig(savepath,transparent = True, dpi = 300, bbox_inches="tight");


# ## Bar plot for excess return

# In[26]:

plt.clf
plt.cla
plt.close('all')

style.use('seaborn-pastel')
# Data to plot
xdata = df.index
ydata0 = df['DoD - FTSE 100 (%)']
# ydata1 = df['PPFTSE 100']
       
# Setting up the fig3ure environment
fig2 = plt.figure(figsize=(20,10))

# Defining the grid and adding plots
ax2 = plt.subplot2grid((1,1),(0,0), facecolor = 'white')
plt.subplots_adjust(left = 0.05, bottom = 0.1, right = .65, top = 0.95, wspace = 0.0, hspace = 0)

ax2.bar(xdata,ydata0, width=20, label = 'Difference')
# ax1.plot_date(xdata, ydata1, '-', label='FTSE 100')

# Plot title, x-label and y-label
plt.ylabel('Difference')
plt.title('DoD - FTSE 100 (%)', loc = 'left', fontweight="bold")

# Adjusting the tickers on x-axis and y-axis
fig2.autofmt_xdate(rotation=90) # To rotate x-ticks
ax2.xaxis.set_major_locator(mdates.YearLocator())

fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
yticks = mtick.FormatStrFormatter(fmt)
ax2.yaxis.set_major_formatter(yticks)


# Adjusting the legend box
# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0 + box.height * 0.1,
#     box.width, box.height * 0.9])
# ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 0),
#     fancybox=True, shadow=True, ncol=8)

# Adjusting the grids
ax2.xaxis.grid(False)
ax2.yaxis.grid(alpha = 1) # how visible is the lines

# plt.tight_layout()

# Saving the fig3ure
savepath = 'C:/Users/Christopher/Dropbox/2. ICMA Centre/PROTEXT test/ICM_296/Excessreturn_bar.png'
plt.savefig(savepath,transparent = True, dpi = 200, bbox_inches="tight");


# ## FTSE 100 Histogram

# In[27]:

plt.clf
plt.cla
plt.close('all')
style.use('bmh')

# Data to plot
y = fama.loc[:,'Rm-Rf']

# n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
n, bins, patches = plt.hist(y, 20, normed=1, label = 'Distribution', rwidth=.75)
plt.axis([np.min(y)*1.2, np.max(y)*1.3, 0, np.max(n)*1.2])
ax = plt.gca()
ax.set_facecolor('white')
y = mlab.normpdf(bins, np.mean(y), np.std(y, ddof=1))
l = plt.plot(bins, y, 'r--', linewidth=1, label ='Best fit')

# Adjusting the grids
plt.grid(False)
ax.legend(loc = 'upper left')
# plt.tight_layout()
#plt.show() # Might not be necessary to display all the fig3ures

#Etc.
plt.xlabel('Excess return')
plt.title('Histogram of FTSE 100 Excess Return', loc = 'left', fontweight = 'bold')
 
# Saving the figure
savepath = 'C:/Users/Christopher/Dropbox/2. ICMA Centre/PROTEXT test/ICM_296/FTSE100Histogram.png'
plt.savefig(savepath,transparent = True, dpi = 300, bbox_inches="tight");


# ## DoD Histogram

# In[28]:

plt.clf
plt.cla
plt.close('all')

# Data to plot
y = fama.loc[:,'DoD-Rf']

# n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
n, bins, patches = plt.hist(y, 20, normed=1, label = 'Distribution', rwidth=0.75)
plt.axis([np.min(y)*1.2, np.max(y)*1.3, 0, np.max(n)*1.2])
ax = plt.gca()
ax.set_facecolor('white')
y = mlab.normpdf(bins, np.mean(y), np.std(y, ddof=1))
l = plt.plot(bins, y, 'r--', linewidth=1, label = 'Best fit')

# Adjusting the grids
plt.grid(False)

ax.legend(loc = 'upper left')
# plt.tight_layout()
#Etc.
plt.xlabel('Excess return')
plt.title('Histogram of DoD Excess Return', loc = 'left', fontweight = 'bold')

# Saving the fig3ure
savepath = 'C:/Users/Christopher/Dropbox/2. ICMA Centre/PROTEXT test/ICM_296/DoDHistogram.png'
plt.savefig(savepath,transparent = True, dpi = 300, bbox_inches="tight");


# ## Rolling STD

# In[29]:

plt.clf
plt.cla
plt.close('all')
style.use('bmh')

# Data to plot
xdata = fama.index
ydata0 = fama['DoD-Rf'].rolling(3).std()
ydata1 = fama['DoD-Rf'].rolling(6).std()
ydata2 = fama['DoD-Rf'].rolling(12).std()    

# Setting up the figure environment
fig5 = plt.figure(figsize=(20,10))

# Defining the grid and adding plots
ax5 = plt.subplot2grid((1,1),(0,0),facecolor='white')
plt.subplots_adjust(left = 0.05, bottom = 0.1, right = .65, top = 0.95, wspace = 0.0, hspace = 0)

ax5.plot_date(xdata, ydata0, '-', label='3-month STD')
ax5.plot_date(xdata, ydata1, '-', label='6-month STD')
ax5.plot_date(xdata, ydata2, '-', label='12-month STD')

plt.plot((np.min(fama.index), np.max(fama.index)), (DoD_excess_std, DoD_excess_std), '--', label ='Constant STD')
plt.plot((np.min(fama.index), np.max(fama.index)), 
         (np.sqrt(semi_variance1), np.sqrt(semi_variance1)), '-.', label ='Constant semi-STD')

# Plot title, x-label and y-label
plt.title('Rolling Standard Deviation', loc = 'left', fontweight = 'bold')

# Adjusting the tickers on x-axis and y-axis
fig5.autofmt_xdate(rotation=90)
ax5.xaxis.set_major_locator(mdates.YearLocator())

# Adjusting the legend box

# Adjusting the grids
ax5.xaxis.grid(False)
ax5.yaxis.grid(alpha = 0) # how visible is the lines

ax5.legend(fontsize = 'large')
ax5.tick_params(axis='x', labelsize=15)
ax5.tick_params(axis='y', labelsize=15)


plt.ylim((0,np.max(ydata0)*1.1))
plt.xlim((xdata.min(), xdata.max()))
# Saving the figure
savepath = 'C:/Users/Christopher/Dropbox/2. ICMA Centre/PROTEXT test/ICM_296/Rolling_std.png'
plt.savefig(savepath,transparent = True, dpi = 300, bbox_inches="tight");


# ## Rolling std2

# In[30]:

plt.clf
plt.cla
plt.close('all')
style.use('bmh')

# Data to plot
xdata = df.index
ydata0 = (df.loc[:,df.columns[0]]/100).rolling(3).std(ddof = 1)
ydata1 = (df.loc[:,df.columns[1]]/100).rolling(3).std(ddof = 1)
      
# Setting up the figure environment
fig2 = plt.figure(figsize=(20,10))

# Defining the grid and adding plots
ax2 = plt.subplot2grid((1,1),(0,0),facecolor='white')
plt.subplots_adjust(left = 0.05, bottom = 0.1, right = .65, top = 0.95, wspace = 0.0, hspace = 0)

ax2.plot_date(xdata, ydata0, '-', label='DoD 3-month Rolling STD')
ax2.plot_date(xdata, ydata1, '-', label='FTSE 100 3-month Rolling STD')

# Plot title, x-label and y-label
# plt.ylabel('Value')
plt.title('Rolling Standard Deviation', loc = 'left', fontweight = 'bold')

# Adjusting the tickers on x-axis and y-axis
fig2.autofmt_xdate(rotation=90)
ax2.xaxis.set_major_locator(mdates.YearLocator())

# Adjusting the grids
ax2.xaxis.grid(False)
ax2.yaxis.grid(alpha = 0) # how visible is the lines

ax2.legend(fontsize = 'large')
ax2.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)


plt.ylim((0, np.max(ydata0)*1.1))

# Saving the figure
savepath = 'C:/Users/Christopher/Dropbox/2. ICMA Centre/PROTEXT test/ICM_296/Rolling_std2.png'
plt.savefig(savepath, transparent = True, dpi = 300, bbox_inches= "tight");


# ## Max Drawdown plot

# In[31]:

plt.clf
plt.cla
plt.close('all')

style.use('fivethirtyeight')

# Data to plot
xdata = df.index
ydata0 = df['PPDoD']
       
# Setting up the figure environment
fig7 = plt.figure(figsize=(20,10))

# Defining the grid and adding plots
ax7 = plt.subplot2grid((1,1),(0,0),facecolor='white')
plt.subplots_adjust(left = 0.05, bottom = 0.1, right = .65, top = 0.95, wspace = 0.0, hspace = 0)

ax7.plot_date(xdata, ydata0, '-', label='DoD Portfolio Value')
plt.plot(('2007-10-31', '2009-02-28'), (df['PPDoD']['2007':'2009'].max(), df['PPDoD']['2007':'2009'].min()), '-', label = 'Max Drawdown Period')
plt.plot(('2007-10-31', '2009-02-28'), (df['PPDoD']['2007':'2009'].min(), df['PPDoD']['2007':'2009'].min()), 'r--')
plt.plot(('2007-10-31', '2007-10-31'), (df['PPDoD']['2007':'2009'].max(), df['PPDoD']['2007':'2009'].min()), 'r--')

# Plot title, x-label and y-label
plt.ylabel('Portfolio Value')
plt.title('Maximum\nDrawdown', loc = 'left', fontweight = 'bold')

# Adjusting the tickers on x-axis and y-axis
fig7.autofmt_xdate(rotation=90)
ax7.xaxis.set_major_locator(mdates.YearLocator())
ax7.axes.get_yaxis().set_major_formatter(
        FuncFormatter(lambda x, p: format(int(x), ',')))

# Adjusting the grids
ax7.xaxis.grid(False)
ax7.yaxis.grid(alpha = 0) # how visible is the lines
ax7.tick_params(axis='x', labelsize=15)
ax7.tick_params(axis='y', labelsize=15)
ax7.legend(loc = 2,fontsize = 'medium')

plt.ylim((0,np.max(ydata0)*1.1))

# Saving the figure
savepath = 'C:/Users/Christopher/Dropbox/2. ICMA Centre/PROTEXT test/ICM_296/Max_drawdown.png'
plt.savefig(savepath,transparent = True, dpi = 300, bbox_inches="tight");


# In[32]:

end = time.time()

print('-------------------------------------'
      '\n{:.4f} seconds used to load script'
      '\n-------------------------------------'
      .format(end-start));

