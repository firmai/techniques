import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, roc_curve
from sklearn.linear_model import LogisticRegression
import clean_data
import pickle

def validation_train_test_split(df):
    ''' Split data into training and testing sets '''
    if 'df' == 'df_small_artists':
        y = df.pop('if_sells_2000')
        X = df
        X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    else:
        y = df.pop('if_sells_20000')
        X = df
        X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    return X_train, y_train, X_test, y_test

def logistic_regression_gridsearch(X_train, y_train, X_test, y_test):
    ''' Logistic regression with grid search '''
    lr_grid = {'C':[0.001, 0.01, 0.1, 1, 10],
        'penalty':['l1','l2']}
    lr = LogisticRegression()
    lr_gridsearch = GridSearchCV(lr,
                          lr_grid,
                          scoring='accuracy',
                          cv=5,
                          n_jobs=-1)
    lr_gridsearch.fit(X_train, y_train)
    lr_best_model = lr_gridsearch.best_estimator_

    lrclf = lr_best_model
    lrclf.fit(X_train, y_train)
    lrclf_predicted = lrclf.predict(X_test)

    lrclf_accuracy = lrclf.score(X_test, y_test)
    print ('Accuracy : {}'.format(lrclf_accuracy))

    lrclf_precision_recall_f1 = precision_recall_fscore_support(y_test, lrclf_predicted, average='binary')
    print ('Precision : {}'.format(lrclf_precision_recall_f1[0]),
      'Recall : {}'.format(lrclf_precision_recall_f1[1]),
      'F1 Score : {}'.format(lrclf_precision_recall_f1[2]))
    lrclf_predicted_proba = lrclf.predict_proba(X_test)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lrclf_predicted_proba[:, -1])

    return fpr_lr, tpr_lr, lrclf_accuracy, lrclf_precision_recall_f1

def random_forest_gridsearch(X_train, y_train, X_test, y_test):
    ''' Random forest classifier with grid search '''
    rf_parameters = {'max_leaf_nodes':[30, 35, 40, 45],
              'n_estimators':[300, 400, 500, 600],
             'max_depth': [8, 10, 15, 20]}
    rf = RandomForestClassifier()

    rf_gridsearch = GridSearchCV(rf,
                             rf_parameters,
                             cv=5,
                             scoring='accuracy',
                             n_jobs=-1)

    rf_gridsearch.fit(X_train, y_train)
    rf_best_model = rf_gridsearch.best_estimator_

    rfclf = rf_best_model
    rfclf = rfclf.fit(X_train, y_train)
    rfclf_predicted = rfclf.predict(X_test)

    rfclf_accuracy = rfclf.score(X_test, y_test)
    print ('Accuracy : {}'.format(rfclf_accuracy))

    rfclf_precision_recall_f1 = precision_recall_fscore_support(y_test, rfclf_predicted, average='binary')
    print ('Precision : {}'.format(rfclf_precision_recall_f1[0]),
      'Recall : {}'.format(rfclf_precision_recall_f1[1]),
      'F1 Score : {}'.format(rfclf_precision_recall_f1[2]))
    rfclf_predicted_proba = rfclf.predict_proba(X_test)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rfclf_predicted_proba[:, -1])

    return fpr_rf, tpr_rf, rfclf_accuracy, rfclf_precision_recall_f1

def random_forest_feature_importance(rfclf):
    ''' Random forest feature importance '''
    feature_scores_rf = pd.DataFrame({'Avg Gini Importance': rfclf.feature_importances_}, index=X_train.columns)
    feature_scores_rf = feature_scores_rf.sort_values(by='Avg Gini Importance')
    feature_scores_rf.iloc[-12:].plot(kind='barh')

def ada_boosting_gridsearch(X_train, y_train, X_test, y_test):
    ''' Adaptive boosting classifier with grid search '''
    ada_boosting_grid = {'learning_rate': [0,1, 0.15, 0.2, 0.25],
                     'n_estimators': [200, 300, 400, 500]}

    adab = AdaBoostClassifier()
    ada_gridsearch = GridSearchCV(adab,
                    ada_boosting_grid,
                    cv=5,
                    n_jobs=-1,
                    scoring='accuracy')

    ada_gridsearch.fit(X_train, y_train)
    best_adaclf_model = ada_gridsearch.best_estimator_

    adabclf = best_adaclf_model
    adabclf.fit(X_train, y_train)
    adabclf_predicted = adabclf.predict(X_test)

    adabclf_accuracy = adabclf.score(X_test, y_test)
    print ('Accuracy : {}'.format(adabclf_accuracy))

    adabclf_precision_recall_f1 = precision_recall_fscore_support(y_test, adabclf_predicted, average='binary')
    print ('Precision : {}'.format(adabclf_precision_recall_f1[0]),
      'Recall : {}'.format(adabclf_precision_recall_f1[1]),
      'F1 Score : {}'.format(adabclf_precision_recall_f1[2]))

    adabclf_predicted_proba = adabclf.predict_proba(X_test)
    fpr_adab, tpr_adab, _ = roc_curve(y_test, adabclf_predicted_proba[:,-1])

    return fpr_adab, tpr_adab, adabclf_accuracy, adabclf_precision_recall_f1

def gradient_boosting_gridsearch(X_train, y_train, X_test, y_test):
    '''Gradient boosting classifier with grid search'''
    gb_grid = {'learning_rate':[0.01, 0.025, 0.05, 0.08],
              'n_estimators':[200, 300, 400, 500],
             'max_depth': [4, 6, 8, 10],
                'subsample': [0.5, 0.7, 0.9]}

    gb = GradientBoostingClassifier()
    gb_gridsearch = GridSearchCV(gb,
                             gb_grid,
                             cv=5,
                             scoring='accuracy',
                             n_jobs=-1)

    gb_gridsearch.fit(X_train, y_train)
    best_gb_model = gb_gridsearch.best_estimator_

    gbclf = best_gb_model
    gbclf.fit(X_train, y_train)
    gbclf_predicted = gbclf.predict(X_test)

    gbclf_accuracy = gbclf.score(X_test, y_test)
    print ('Accuracy : {}'.format(gbclf_accuracy))

    gbclf_precision_recall_f1 = precision_recall_fscore_support(y_test, gbclf_predicted)
    print ('Precision : {}'.format(gbclf_precision_recall_f1[0][1]),
      'Recall : {}'.format(gbclf_precision_recall_f1[1][1]),
      'F1 Score : {}'.format(gbclf_precision_recall_f1[2][1]))

    gbclf_predicted_proba = gbclf.predict_proba(X_test)
    fpr_gb, tpr_gb, _ = roc_curve(y_test, gbclf_predicted_proba[:, -1])

    return fpr_gb, tpr_gb, gbclf_accuracy, gbclf_precision_recall_f1

def ensemble_models(X_train, y_train, X_test, y_test):
    '''Ensemble of random forest, gradient boosting, adaboosting models, using voting classifier'''
    ensembleclf = VotingClassifier(estimators=[('rf', rfclf), ('gdb', gbclf),('ada', adabclf)], voting='soft')
    ensembleclf.fit(X_train, y_train)
    ensemble_predicted = ensembleclf.predict(X_test)

    ensemble_accuracy = ensembleclf.score(X_test, y_test)
    print ('Accuracy : {}'.format(ensemble_accuracy))
    ensembleclf_precision_recall_f1 = precision_recall_fscore_support(y_test, ensemble_predicted, average='binary')
    print ('Precision : {}'.format(ensembleclf_precision_recall_f1[0]),
      'Recall : {}'.format(ensembleclf_precision_recall_f1[1]),
      'F1 Score : {}'.format(ensembleclf_precision_recall_f1[2]))

    ensembleclf_predicted_prob = ensembleclf.predict_proba(X_test)
    fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, ensembleclf_predicted_prob[:, -1])

    return fpr_ensemble, tpr_ensemble, ensemble_accuracy, ensembleclf_precision_recall_f1
