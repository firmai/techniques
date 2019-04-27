# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:42:04 2017

@author: Christopher
"""
import pandas as pd

path = "Pickles/"

totret = pd.read_pickle(path+'totret')
icm = pd.read_pickle(path+'icm')
dividends = pd.read_pickle(path+'dividends')
dividends2 = pd.read_pickle(path+'dividends2')
investmentuniverse = pd.read_pickle(path+'investmentuniverse')
rf = pd.read_pickle(path+'rf')
df = pd.read_pickle(path+'df')
fama = pd.read_pickle(path+'fama')
DoDportfolios = pd.read_pickle(path+'DoDportfolios')
