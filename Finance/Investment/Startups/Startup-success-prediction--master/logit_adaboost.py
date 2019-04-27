# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:39:50 2017

@author: medha
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

# load dataframe. shape = (447, 19)
df = pd.read_csv('after_chi_sqr_0.01.csv')

# take required columns
df = df.iloc[:, np.array([-1,0,1,3,4,10,13,16]) + 1]
print df.columns

# convert target to 0 and 1s
df['Dependent.Company.Status'] = pd.DataFrame(label_binarize(df['Dependent.Company.Status'], classes = ['FAILED','SUCCESS']))

# training data with 115 success. 115 failed
train = df[df['Dependent.Company.Status'] == 1].head(115)
train = train.append(df[df['Dependent.Company.Status'] == 0].head(115))

# take rest as testing data
test = pd.concat([df, train])
test = test.drop_duplicates(keep = False)

# drop company name column
train = train.drop('Company_Name', axis = 1)
test = test.drop('Company_Name', axis = 1)

# create x and y for trainig and testing
# sklearn needs categorical features as numbers, hence convert
x_train, y_train = train.ix[:, 1:], train.ix[:, 0]
x_test, y_test = test.ix[:, 1:], test.ix[:, 0]
x_train.iloc[:, 1:] = x_train.iloc[:, 1:].apply(LabelEncoder().fit_transform)
x_test.iloc[:, 1:] = x_test.iloc[:, 1:].apply(LabelEncoder().fit_transform)

# printing first few tuples of x and y of training dataset
print x_train.head(50)
print y_train.head(50)

#apply logistic regression
logit = LogisticRegression()
logit.fit(x_train, y_train)

# print different accuracy measurements
print 'accuracy score: ', logit.score(x_test, y_test)
print 'precision:', precision_score(y_test, logit.predict(x_test), average='weighted')
print 'recall:', recall_score(y_test, logit.predict(x_test), average='weighted')
print 'mean cross validation score:', np.mean(cross_val_score(logit, pd.concat([x_train, x_test]), pd.concat([y_train, y_test])))

# apply adaboost for no of estimators 1 to 20 and print accuracies
for i in range(1, 20):
    clf = AdaBoostClassifier(n_estimators= i, base_estimator=logit)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    precision = precision_score(y_test, clf.predict(x_test), average='weighted')
    recall = recall_score(y_test, clf.predict(x_test), average='weighted')
    cross_val_mean = np.mean(cross_val_score(clf, pd.concat([x_train, x_test]), pd.concat([y_train, y_test])))
    print i, score , precision, recall, cross_val_mean
    
# initial values accuracy, precision, recall, cross validation mean
# 0.741935483871 0.814616332408 0.741935483871 0.724832214765

# after boosting

# maximum accuracy, precision, recall at no of estimators = 10
# 10 0.755760368664 0.818101492867 0.755760368664 0.733780760626
    
#maximum mean cross validation score at no of estimators = 19
# 19 0.746543778802 0.815758972211 0.746543778802 0.744966442953

