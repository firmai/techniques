"""
Model Selection
---------------

Evaluate clasification models through exhaustive grid search, stratified 
5-fold cross-validation, and Area Under Precision-Recall Curve (AUPRC) scorer.

@author: Rodrigo Santis
"""
print(__doc__)

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import time
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, make_scorer
from sklearn import linear_model, tree, ensemble
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from ensampling.bagging import Blagging

n_runs=30 # Define the number of models to be trained
scorer = make_scorer(average_precision_score, needs_threshold=True, average="micro",)#make_scorer(cohen_kappa_score)#'roc_auc' 

min_samples_leaf=5
n_estimators=10
criterion='entropy'
max_depth=np.arange(3,45,5)
max_depth=[3,4,5,7,10,15,20,30,50]
dataset='kaggle' #'data/bopredict.csv'
n_folds=5
save_run=10

df = pd.read_csv('data/'+dataset+'.csv')
X = df.drop(['went_on_backorder','sku'],axis=1).values
y = df['went_on_backorder'].values

print("dataset:",dataset)

estimators = [
    ("Logistic Regression", 'lgst', 
    linear_model.LogisticRegression(), 
    {'C':np.logspace(0,3,4),
     'penalty':['l1','l2'],
    }),   
        
    ("Decision Tree", 'cart',
     tree.DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,
                                 criterion=criterion),
    {'max_depth':max_depth,
     #'max_features':[3,5,10,None],
     #'splitter':['best','random'],
     'criterion':['entropy','gini'],
    }),

    ("RandomUnderSampling", 'rus',
     Pipeline([('res', RandomUnderSampler()),
               ('tree', tree.DecisionTreeClassifier(
                       min_samples_leaf=min_samples_leaf, criterion=criterion))
               ]),
    {'tree__max_depth':max_depth,
    }),

    ("SMOTE", 'smt',
     Pipeline([('res', SMOTE()),
               ('tree', tree.DecisionTreeClassifier(
                       min_samples_leaf=min_samples_leaf, criterion=criterion))
               ]),
    {'tree__max_depth':max_depth,
    }),

    ("UnderBagging", 'ub',
    Blagging(n_estimators=n_estimators,
    base_estimator=tree.DecisionTreeClassifier(
            min_samples_leaf=min_samples_leaf,criterion=criterion)),
    {'max_depth':max_depth,
     }),

    ("RandomForest", "rf",
     ensemble.RandomForestClassifier(n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf, criterion=criterion),
    {'max_depth':max_depth,
    }),

    ("GradientBoosting", "gb",
     ensemble.GradientBoostingClassifier(n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf),
    {'max_depth':[10,],
    }),
]

for est_full_name, est_name, est, params in estimators:
    print ('\n%s\n%s\n' % ('-'*25, est_full_name))
    print ('Run\tEst\tScore\tAUROC\tAUPRC\tTime\tBest parameters')
    matriz = []
    t0 = time.time()
    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                    test_size=0.15, random_state=run)
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(run*9))
        gs = GridSearchCV(est, params, cv=kf,# n_iter=n_iter_search,
                          scoring=scorer, verbose=0,n_jobs=-1)  
        t1 = time.time()
        gs.fit(X_train, y_train)
 
        y_prob0 = gs.best_estimator_.predict_proba(X_train)[:,1]
        y_prob = gs.best_estimator_.predict_proba(X_test)[:,1]
        
        roc =  roc_auc_score(y_test, y_prob)
        pr = average_precision_score(y_test, y_prob)   
        
        run_time = time.time() - t1
        avg_time = run_time/gs.n_splits_
        
        print ("%i\t%s\t%.4f\t%.4f\t%.4f\t%.2f\t%s" % (run, est_name, 
            gs.best_score_, roc, pr, avg_time, gs.best_params_))

        
        # get importance
        imp = []
        mdl = gs.best_estimator_
        if est_name in ['ub','sbag']:
            imp = np.mean([
                    e.feature_importances_ for e in mdl.estimators_
                    ], axis=0)
        elif est_name in ['rus','smt']:
            imp = mdl.named_steps['tree'].feature_importances_
        elif est_name == 'lgst':
            imp = mdl.coef_.ravel()
        else:
            imp = mdl.feature_importances_
        
        matriz.append(
        {   'run'           : run,
            'estimator'     : est_name,         
            'roc'           : roc,
            'pr'            : pr,
            'best_params'   : gs.best_params_, 
            'avg_time'      : avg_time,
            'importance'    : imp,
        })
        
        if run == save_run:
            path = 'results/pkl/'+dataset+'/'+est_name.lower() + '.pkl'        
            joblib.dump(gs.best_estimator_, path) 
 
    print("Elapsed time: %0.3fs" % (time.time()-t0))
    # Save results
    data = pd.DataFrame(matriz)
    