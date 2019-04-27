"""
Results
--------------------------------------------------

@author: Rodrigo Santis
"""
print(__doc__)

import numpy as np
import pandas as pd # data processing, CSV file I/O
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import itertools
sns.set_style('white')
ls = itertools.cycle(['-','--','-.',':'])

n_run=10 # desired run
max_run=30 # total runs
dataset = 'kaggle'
format = 'jpeg'

estimators = ['lgst','cart','rus', 'smt','rf','gb','ub']

def roc_plot(estimators, models):
    """ Plot ROC curves for each estimator.
    """
    f, ax = plt.subplots()
    for est, mdl in zip(estimators, models):
        _roc_plot(y_test,mdl.predict_proba(X_test),label=est,ax=ax,l=next(ls))
    ax.plot([0,1], [0,1], color='k', linewidth=0.5, linestyle='--', 
        label='Random Classifier')    
    ax.legend(loc="lower right")    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(False)
    sns.despine()
    plt.savefig('img/'+dataset+'/auc_score.'+format,format=format,bbox_inches='tight',dpi=450)
    plt.show()

def _roc_plot(y_true, y_proba, label=' ', l='-', lw=1.0, ax=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
    ax.plot(fpr, tpr, linestyle=l, linewidth=lw,
            label="%s (area=%.3f)"%(label,roc_auc_score(y_true, y_proba[:,1])))

def pr_plot(estimators, models):
    """ Plot Precision-Recall curves for each estimator.
    """
    f, ax = plt.subplots()
    for est, mdl in zip(estimators, models):
        _pr_aux(y_test,mdl.predict_proba(X_test),label=est,ax=ax,l=next(ls))
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    sns.despine()
    plt.savefig('img/'+dataset+'/precision_recall.'+format,format=format,bbox_inches='tight',dpi=450)
    plt.show()    

def _pr_aux(y_true, y_proba, label=' ', l='-', lw=1.0, ax=None):
    precision, recall, _ = precision_recall_curve(y_test,
                                                  y_proba[:,1])
    average_precision = average_precision_score(y_test, y_proba[:,1],
                                                     average="micro")
    ax.plot(recall, precision, label='%s (area=%.3f)'%(label,average_precision),
            linestyle=l, linewidth=lw)

def show_score(estimators):
    """ Plot ROC curves for each estimator.
    """
    results = pd.DataFrame([],columns=['run','estimator','roc','pr','best_params','avg_time'])
    for est in estimators:    
        df = pd.read_pickle('results/pkl/'+dataset+'/'+est+'_results.pkl').drop('run',axis=1)
        results = results.append(df)
    summary = results.groupby('estimator').agg({'roc':['mean','std'],
                    'pr':['mean','std'],'avg_time':['mean','std']})
    params = results.groupby('estimator').agg({'best_params':lambda x:
                                stats.mode(x.astype(str))[0]})
    print("%s\n\n%s" % (summary.round(4), params))
    return results, summary

# Load data
df = pd.read_csv('data/'+dataset+'.csv')
X = df.drop(['went_on_backorder','sku'],axis=1).values
y = df['went_on_backorder'].values

# Split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, 
                                    test_size=0.15, random_state=n_run)

# Load pkl
models = []
for est in estimators:
    models.append(joblib.load('results/pkl/'+dataset+'/'+est+'.pkl'))

print ('\n%s\n%s\n' % ('Scores by estimator and best parameters', '-'*25))
results, summary = show_score(estimators)
summary.round(4).to_csv('img/'+dataset+'/results.txt', sep=' ', mode='a')

print ('\n%s\n%s\n' % ('Area under ROC curve', '-'*25))
roc_plot(estimators, models)

print ('\n%s\n%s\n' % ('Precision-recall curve', '-'*25))
pr_plot(estimators, models)

print ('\n%s\n%s\n' % ('PR/ROC trade-off', '-'*25))
pr = summary['pr']['mean'].values
roc = summary['roc']['mean'].values
names = summary.index.values
f, ax = plt.subplots()
plt.errorbar(pr, roc,xerr=summary['pr']['std'],ecolor='0.25',elinewidth=1,mfc='k',
             mec='k',yerr=summary['roc']['std'],fmt='o',capthick=0,capsize=3,
             markersize=5, linestyle=None)
for s, x, y in zip(names,pr,roc):
        plt.text(x+0.003,y-0.0015,s.lower(),fontsize=13)
ax.set_xlabel('AUPRC')
ax.set_ylabel('AUROC')
ax.grid(False)
sns.despine()
plt.savefig('img/'+dataset+'/errobar.'+format,format=format,bbox_inches='tight',dpi=450)
plt.show()

m3_pr = pr
m3_roc = roc

print ('\n%s\n%s\n' % ('Feature importance', '-'*25))
for est in estimators:
    df_imp = pd.read_pickle('results/pkl/'+dataset+'/'+est+'_results.pkl').drop('run',axis=1)
    imp=[]
    feat=[]
    for row in df_imp['importance'].values:
        imp.extend(list(row))
        feat.extend(list(df.drop(['went_on_backorder','sku'],axis=1).columns.values))
    importance = pd.DataFrame(data={'features':feat, 'importance':imp,})
    
    f, ax = plt.subplots()
    ax = sns.barplot(y='features',x='importance',palette="Blues_r",data=importance)
    #ax.legend(loc="lower right")
    plt.title(est)
    plt.savefig('img/'+dataset+'/importance_'+est+'.'+format,format=format,bbox_inches='tight',dpi=450)
    plt.show()
