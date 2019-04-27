"""
Preprocessing
-------------

This script processes the original dataset, transforming the original attributes
into numeric attributes and saving into a smaller and consolidated file.

@author: Rodrigo Santis
"""
print(__doc__)

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt

def process(df):
    """
    Some strategies adopted:
    - Binaries were converted from strings ('Yes' and 'No') to 1 and 0.
    - The attributes related to quantities were normalized (std dev equal to 1)
    per row. Therefore, parts with different order of magnitudes are 
    approximated. For example: 1 unit of a expensive machine may be different 
    from 1 unit of a screw, but if we standard deviate all the quantities we 
    have, we can get a better proportion of equivalence between those items.
    - Missing values for lead_time and perf_month_avg were replaced using 
    series median and mean. 
    """
    # Imput missing lines and drop line with problem
    from sklearn.preprocessing import Imputer
    df['lead_time'] = Imputer(strategy='median').fit_transform(
                                    df['lead_time'].values.reshape(-1, 1))
    df = df.dropna()
    for col in ['perf_6_month_avg', 'perf_12_month_avg']:
        df[col] = Imputer(missing_values=-99).fit_transform(
                                    df[col].values.reshape(-1, 1))
    # Convert to binaries
    for col in ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk',
               'stop_auto_buy', 'rev_stop', 'went_on_backorder']:
        df[col] = (df[col] == 'Yes').astype(int)
    # Normalization    
    from sklearn.preprocessing import normalize
    qty_related = ['national_inv', 'in_transit_qty', 'forecast_3_month', 
                   'forecast_6_month', 'forecast_9_month', 'min_bank',
                   'local_bo_qty', 'pieces_past_due', 'sales_1_month', 
                   'sales_3_month', 'sales_6_month', 'sales_9_month',]
    df[qty_related] = normalize(df[qty_related], axis=1)
    return df

def plot_2d(X, y, title=''):
    """
    Plot the two major components using PCA, giving a general interpretation of
    the dataset.
    """
    from sklearn.preprocessing import StandardScaler
    X_std = StandardScaler().fit_transform(X)

    from sklearn.decomposition import PCA
    dec = PCA(n_components=2)
    X_reduced = dec.fit_transform(X_std)
    
    f, ax = plt.subplots(figsize=(6,6))
    ax.scatter(X_reduced[y==0,0], X_reduced[y==0,1], 
               facecolors='none', edgecolors='0.75', label="Negative")
    ax.scatter(X_reduced[y==1,0], X_reduced[y==1,1], c='0.25', marker='*', 
               label='Positive')
    ax.legend(loc='lower left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    print ("Explained variance ratio: %.2f%%" % 
           (100*dec.explained_variance_ratio_.sum()))
    print (dec.explained_variance_ratio_)
    plt.savefig('img/plot2d.jpeg',format='jpeg',bbox_inches='tight',dpi=450)
    plt.show()

# Load files
cols=range(0,23)
train = pd.read_csv('data/kaggle/Kaggle_Training_Dataset_v2.csv', usecols=cols)
test = pd.read_csv('data/kaggle/Kaggle_Test_Dataset_v2.csv', usecols=cols)
df = process(train.append(test))

# Sampling
sample = df.sample(5000, random_state=36)
X_sample = sample.drop('went_on_backorder',axis=1).values
y_sample = sample['went_on_backorder'].values

plot_2d(X_sample, y_sample)

# Save new file
df.round(6).to_csv('data/kaggle.csv',index=False)