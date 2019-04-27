# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:28:20 2017

@author: Christopher
"""

#Preamp

import pandas as pd
import time
# # Importing data

# In[82]:

path= 'C:/Users/Christopher/Dropbox/2. ICMA Centre/3. Module/Submission/ICM296 data.xlsm'

start = time.time()
icm = pd.read_excel(path,parse_cols='A:HD',index_col=0, header=2, sheetname='Stock Close Prices')
totret = pd.read_excel(path,parse_cols='A:HD',index_col=0, header=2, sheetname='Total return')
dividends = pd.read_excel(path, header = 3, parse_cols= 'A:HD', sheetname='Pivot', index_col=0)
dividends2 = pd.read_excel(path, header = 3, parse_cols= 'A:HD', sheetname='Pivot2', index_col=0)
investmentuniverse = pd.read_excel(path,parse_cols='B:P', header=2, sheetname='FTSE100 History')
rf = pd.read_excel(path,parse_cols='H,I',index_col=0, header=1, sheetname='Risk-free rates')
df = pd.read_excel('C:/Users/Christopher/Dropbox/2. ICMA Centre/3. Module/Submission/All portfolio values_2.xlsx', sheetname= 'Sheet1')
fama = pd.read_excel(path,parse_cols='B:D', header=1, sheetname='FamaFrench')
DoDportfolios = pd.read_excel('C:/Users/Christopher/Dropbox/2. ICMA Centre/3. Module/Submission/ICM296 data.xlsm', 
                              sheetname='Final Portfolios (2)', 
                              parse_cols='AA:AO', header = 0)



# # To pickles
totret.to_pickle('totret')
icm.to_pickle('icm')
dividends.to_pickle('dividends')
dividends2.to_pickle('dividends2')
investmentuniverse.to_pickle('investmentuniverse')
rf.to_pickle('rf')
df.to_pickle('df')
fama.to_pickle('fama')
DoDportfolios.to_pickle('C:/Users/Christopher/DoDportfolios')

end = time.time()
print(end-start)

