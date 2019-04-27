###############################
######SIOP ML Competition######
###########TEAMDDI#############
###########Feb 2018############
###############################

################################################################
####### Note: This syntax starts after the external data #######
############ has been merged with the original data ############
################# See Part I for more details ##################
################################################################


### IMPORT LIBRARIES ###
import numpy as np
import pandas as pd
import gc
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


### DATA CREATION ###

# v1: original data + external data + var_miss (number of missing vars)

# Importe full merged dataset (includes training and testing data)
Full = pd.read_csv('Merged.csv', index_col = None)
X = Full.iloc[:, 6:] # 180 vars

# Recode selected categorical vars to numeric vars
X['X_Gender'] = X['X_Gender'].replace(['Female','Male'], [0,1])
X['S_Gender2009'] = X['S_Gender2009'].replace(['Female','Male'], [0,1])
X['S_Gender2008'] = X['S_Gender2008'].replace(['Female','Male'], [0,1])
X['S_Gender2007'] = X['S_Gender2007'].replace(['Female','Male'], [0,1])
X['S_Gender2006'] = X['S_Gender2006'].replace(['Female','Male'], [0,1])
X['S_Gender2005'] = X['S_Gender2005'].replace(['Female','Male'], [0,1])
X['S_Gender2004'] = X['S_Gender2004'].replace(['Female','Male'], [0,1])
X['X_PotentialLevel_TooEarly'] = np.where(X['X_PotentialLevel'] == 'Too Early', 1, 0)
X['X_PotentialLevel'] = X['X_PotentialLevel'].replace(['M1','M2','M3','M4','M5','M6','M7+','Non-M','Too Early'], [1,2,3,4,5,6,7,8,np.nan])
X['X_ExpOutsideHomeCountry'] = X['X_ExpOutsideHomeCountry'].replace(['No','Yes'], [0,1])
X['X_CrossFunctionalExperience'] = X['X_CrossFunctionalExperience'].replace(['No','Yes'], [0,1])
X['X_GLDP'] = X['X_GLDP'].replace(['No','Yes'], [0,1])

# Create var_miss var - count of missing vars
# var_miss is correlated w/ turnover
X['var_miss'] = X.isnull().sum(axis=1)


# Create a sub-dataset for categorical vars that cannot be directly transformed into numeric vars (e.g., country)
# 43 vars
X_cat = pd.concat([X.iloc[:, 2], X.iloc[:, 4:34], X.iloc[:, 85:91], X.iloc[:, 103:109]], axis=1)    

# Change each column to category type
# This is necessary because some categorical vars have numeric values (e.g., supervisorID)
X_cat_col_names = list(X_cat)
# print(X_cat_col_names)

for col in X_cat_col_names:
    X_cat[col] = X_cat[col].astype('category',copy=False)
# print(X_cat.dtypes)    
    
# Create dummy variables for categorical var data
X_cat_dummy = pd.get_dummies(X_cat)
# print(list(X_cat_dummy))

# Keep X variables w/ frequency>10 (this cutoff is arbitrary)
# Number of vars decreased from 55043 to 3453
X_cat_dummy = X_cat_dummy[X_cat_dummy.columns[X_cat_dummy.sum()>10]]


# Create a sub-dataset for numeric variables only
# 139 vars
X_num = pd.concat([X.iloc[:, 0:2], X.iloc[:, 3], X.iloc[:, 34:85], 
                   X.iloc[:, 91:103], X.iloc[:, 109:182]], axis=1)
# print(X_num.dtypes)

# Replace missing data with column means
X_num = X_num.fillna(X_num.mean())

# Combine numeric dataset and categorical dataset
X_clean = pd.concat([X_num, X_cat_dummy], axis=1)

# Save data X_clean as X_clean_v1
X_clean_v1 = X_clean
# X_clean.to_csv('X_clean_v1.csv', encoding='utf-8', index = False)



### FEATURE SELECTION ###

# Extract Y_train from original training data
training = pd.read_csv('TrainingSet.csv')
Y_train = pd.Series(training.iloc[:, 2].values)

# Extract X_train from X_clean
X_train = pd.DataFrame(X_clean.iloc[0:24205, ].values, columns=X_clean.columns)
X_test = pd.DataFrame(X_clean.iloc[24205:, ].values, columns=X_clean.columns)


# Feature scaling
# This is a standard procedure to ensure that the vars are standardized
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.fit_transform(X_test)

# Append indices
X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_clean.columns)
X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_clean.columns)

# Shuffle X_train and Y_train
# This is for proper ML model learning
X_train = X_train.sample(frac=1, random_state=0)
Y_train = Y_train.sample(frac=1, random_state=0)


# Data v1_1 and v1_2: with feature selection based on Lasso
# The goal is to keep the more important features for prediction

# v1_1: Lasso w/ alpha of 0.0001 (adjusting the alpha level to keep more/fewer features)
# Selected 2908 features from 3592 features
clf_v1_1 = linear_model.Lasso(alpha=0.0001, max_iter=2000, random_state=0)
model_v1_1 = SelectFromModel(clf_v1_1)
model_v1_1.fit(X_train, Y_train)

# Save new data as X_clean_v1_1
X_clean_v1_1 = model_v1_1.transform(X_clean)
# X_clean_v1_1.shape

# Append new feature list as column names
feature_list_v1_1 = model_v1_1.get_support()
feature_list_v1_1 = pd.concat([pd.Series(list(X_train)), pd.Series(feature_list_v1_1)], axis=1)
feature_selected_v1_1 = [item[0] for item in feature_list_v1_1.values.tolist() if item[1]]
X_clean_v1_1 = pd.DataFrame(X_clean_v1_1, columns=feature_selected_v1_1)
# X_clean_v1_1.to_csv('X_clean_v1.1.csv', encoding='utf-8', index = False)


# v1_2: Lasso w/ alpha of 0.0005 (adjusting the alpha level to keep more/fewer features)
# Selected 1916 features from 3592 features
clf_v1_2 = linear_model.Lasso(alpha=0.0005, max_iter=1000, random_state=0)
model_v1_2 = SelectFromModel(clf_v1_2)
model_v1_2.fit(X_train, Y_train)

# Save new data as X_clean_v1_2
X_clean_v1_2 = model_v1_2.transform(X_clean)
# X_clean_v1_2.shape

# Append new feature list as column names
feature_list_v1_2 = model_v1_2.get_support()
feature_list_v1_2 = pd.concat([pd.Series(list(X_train)), pd.Series(feature_list_v1_2)], axis=1)
feature_selected_v1_2 = [item[0] for item in feature_list_v1_2.values.tolist() if item[1]]
X_clean_v1_2 = pd.DataFrame(X_clean_v1_2, columns=feature_selected_v1_2)
# X_clean_v1_2.to_csv('X_clean_v1.2.csv', encoding='utf-8', index = False)


# Create data v2 by eliminating Supervisor ID and City
# Those vars have little to no relations w/ turnover
# 1050 vars
X_clean_v2 = pd.concat([X_clean_v1.iloc[:, 0:485], X_clean_v1.iloc[:, 1671:2194], 
                        X_clean_v1.iloc[:, 3550:3592]], axis=1)

    
# Create X_clean_tuples that combines v2 and tuples
# Tuples [n*(n-1)/2] of the top features (highest rs w/ turnover)
X_tuples =  X_clean_v2[['var_miss',
                        'X_TenureDays2009',
                        'X_Age2009',
                        'X_Country2009_China',
                        'TotalUnemp_2009',
                        'X_JobFunction2009_SALES/MARKETING',
                        'X_JobType2009_SALES REPRESENTATIVE',
                        'X_JobSubFunction2009_SALES - PHARMA',
                        'X_Country2004_United States of America',
                        'NumberofSupervisors',
                        'CLI_2009',
                        'X_JobFunction2004_MANUFACTURING',
                        'S_RaceEthnicity2004_White/Caucasian',
                        'X_Country2009_India',
                        'X_RaceEthnicity_White/Caucasian',
                        'X_JobType2004_PRODUCTION OPERATOR',
                        'X_JobFunction2005_SCIENCE AND TECHNOLOGY',
                        'S_PayGradeLevel2009',
                        'EveraSupervisor',
                        'X_Country2005_France',
                        'X_Country2009_Brazil',
                        'CCI_2008',
                        'X_JobType2007_TECHNICIAN',
                        'CCI_2007',
                        'X_OverallPerformanceRating2009',
                        'X_JobSubFunction2004_FILL/FINISH MANUFACTURING OPERATIONS',
                        'TotalUnemp_2008',
                        'X_JobSubFunction2004_ENGINEERING',
                        'X_JobFunction2004_QUALITY ASSURANCE AND CONTROL',
                        'X_Country2009_Australia',
                        'X_PMB_Action_2009',
                        'X_JobType2004_BUSINESS ASSOCIATE',
                        'CCI_2005',
                        'X_JobSubFunction2009_BULK MANUFACTURING OPERATIONS',
                        'X_PMB_Teamwork_2009',
                        'X_PMB_Engagement_2009',
                        'CCI_2006',
                        'X_JobSubFunction2005_DISCOVERY RESEARCH/RESEARCH TECHNOLOGIES',
                        'X_PMB_Accountability_2009',
                        'CLI_2006',
                        'X_JobFunction2004_INFORMATION TECHNOLOGY',
                        'X_JobSubFunction2004_IT BUSINESS INTEGRATION/AFF',
                        'X_Country2007_Italy',
                        'X_PMB_AnticipateChange_2008',
                        'X_Country2004_Spain',
                        'X_JobType2007_CHEMIST',
                        'X_Country2008_Saudia Arabia',
                        'X_JobSubFunction2004_QUALITY CONTROL',
                        'X_Country2009_Malaysia',
                        'X_PMB_EvaluateAct_2008',
                        'X_JobType2006_CLERICAL ASSISTANT',
                        'S_PayGradeLevel2007',
                        'X_PMB_AchieveResultsPeople_2008',
                        'X_PMB_Values_2009',
                        'X_Country2009_Russia',
                        'X_JobSubFunction2009_MAINTENANCE',
                        'X_Country2007_Puerto Rico',
                        'X_JobType2004_CLERICAL ASSISTANT',
                        'X_JobType2005_MANAGEMENT (G-LEVEL)',
                        'X_Country2004_Germany',
                        'S_PayGradeLevel2008',
                        'X_PMB_ImplementQuality_2008',
                        'X_JobSubFunction2008_PRODUCT/PROCESS DEVELOPMENT',
                        'S_RaceEthnicity2007_Hispanic/Latino'
                        ]]

def create_tuples(X):
    # logger.debug("creating feature tuples")
    cols = []
    for i in range(X.shape[1]):
        for j in range((i+1), X.shape[1]):
            cols.append(X.iloc[:, i] + X.iloc[:, j]) 
    return np.vstack(cols).T

X_tuples_t = create_tuples(X_tuples)    
X_clean_tuples = pd.concat([X_clean_v2, pd.DataFrame(X_tuples_t)], axis=1)
# X_clean_tuples.to_csv('X_clean_tuples.csv', encoding='utf-8', index = False)
# 3066 vars


# Create X_clean_triples that combines v2 and triples
# Triples [n*(n-1)*(n-2)/(3*2*1)]  of the top features (highest rs w/ turnover)
X_triples = X_clean_v2[['var_miss',
                        'X_TenureDays2009',
                        'X_Age2009',
                        'X_Country2009_China',
                        'TotalUnemp_2009',
                        'X_JobFunction2009_SALES/MARKETING',
                        'X_Country2004_United States of America',
                        'NumberofSupervisors',
                        'CLI_2009',
                        'X_JobFunction2004_MANUFACTURING',
                        'S_RaceEthnicity2004_White/Caucasian',
                        'X_Country2009_India',
                        'X_JobType2004_PRODUCTION OPERATOR',
                        'X_JobFunction2005_SCIENCE AND TECHNOLOGY',
                        'S_PayGradeLevel2009',
                        'EveraSupervisor',
                        'X_Country2005_France',
                        'X_Country2009_Brazil',
                        'CCI_2008',
                        'X_JobType2007_TECHNICIAN',
                        'X_OverallPerformanceRating2009']]

def create_triples(X):
    # logger.debug("creating feature tuples")
    cols = []
    for i in range(X.shape[1]):
        for j in range((i+1), X.shape[1]):
            for k in range((j+1), X.shape[1]):
                cols.append(X.iloc[:, i] + X.iloc[:, j] + X.iloc[:, k]) #*3571
    return np.vstack(cols).T

X_triples_t = create_triples(X_triples)
X_clean_triples = pd.concat([X_clean_v2, pd.DataFrame(X_triples_t)], axis=1)
# X_clean_triples.to_csv('X_clean_triples.csv', encoding='utf-8', index = False)
# 2380 vars

# Free up some memory
del Full, X, X_cat, X_cat_col_names, X_cat_dummy, X_clean, X_num, X_test, X_test_scaled
del X_train, X_train_scaled, X_triples, X_triples_t, X_tuples, X_tuples_t, Y_train, col
del feature_list_v1_1, feature_list_v1_2, feature_selected_v1_1, feature_selected_v1_2
gc.collect()


### DATA PREPARATION FOR ML ###
# Now that all datasets are created
# let's prepare these datasets for machine learning

# Naming convention: variable_data
# e.g., x_full_train: X variables from the 90% training data
# e.g., x_true_test: X variables from the (true) holdout data

# Extract y_full_train from original training data
# training = pd.read_csv('TrainingSet.csv')
y_full_train = pd.Series(training.iloc[:, 2].values)

# Extract global ID for the true TestSet
# This will be used for the final submission
testing = pd.read_csv('TestSet.csv')
x_true_test_Global_ID = testing.iloc[:, 0]

# Create a list of all cleaned X datasets
l_x_full_clean = [X_clean_v1, X_clean_v1_1, X_clean_v1_2, X_clean_v2, X_clean_tuples, X_clean_triples]

# Extract x_training data from full x data
l_x_full_train = []
l_x_true_test = []
for x_full_clean in l_x_full_clean:
    l_x_full_train.append(pd.DataFrame(x_full_clean.iloc[0:24205, ].values, columns=x_full_clean.columns))
    l_x_true_test.append(pd.DataFrame(x_full_clean.iloc[24205:, ].values, columns=x_full_clean.columns))


# Feature scaling & append indices
sc = StandardScaler()
for i in range(len(l_x_full_train)):
    x_full_train, x_true_test = l_x_full_train[i], l_x_true_test[i]
    x_full_clean = l_x_full_clean[i]
    l_x_full_train[i] = pd.DataFrame(sc.fit_transform(x_full_train), index=x_full_train.index, columns=x_full_clean.columns)
    l_x_true_test[i] = pd.DataFrame(sc.fit_transform(x_true_test), index=x_true_test.index, columns=x_full_clean.columns)

# Shuffle x_full_train and y_full_train for proper model learning
for i in range(len(l_x_full_train)): 
    l_x_full_train[i] = l_x_full_train[i].sample(frac=1, random_state=0)

y_full_train = y_full_train.sample(frac=1, random_state=0)


# Free up some memory
del testing, training, l_x_full_clean, X_clean_v1, X_clean_v1_1
del X_clean_v1_2, X_clean_v2, X_clean_tuples, X_clean_triples
gc.collect()


'''
### HYPERPARAMETER TUNING ###
# Parameter tuning is commented out because it follows an iterative process
# and needs to be done "manually" to identify the optimal parameters

# I focused on three parameters: learning_rate, n_estimators, and max_depth
# for the xgboost models

# Parameter tuning follows an interative process where you hold 
# all other parameters constant while using GridSearchCV to 
# identify the best parameter(s) for the targeted models

# Grid Search is an approach to parameter tuning that will methodically build and 
# evaluate a model for each combination of parameters specified in a grid.

# e.g., the following syntax holds all other parameters constant 
#       while searching for the optimal learning_rate & n_estimators

# After identifying the best parameters in the specified ranges,
# you can adjust the range to search for more optimal/different parameters

param_grid = {'learning_rate':[0.03,0.05],
              'n_estimators':[200,250,300], 
              #'max_depth':[7,10], 
              #'min_child_weight':[3,5,7]
              #'gamma':[4,6,8], 
              #'subsample':[0.5,0.6,0.7],
              #'colsample_bytree':[0.3,0.4,0.5]
              } 

classifier = XGBClassifier(#learning_rate =0.03, 
                           #n_estimators=200, 
                           max_depth=7,
                           min_child_weight=7, 
                           gamma=6, 
                           subsample=0.7, 
                           colsample_bytree=0.4,
                           objective= 'binary:logistic', 
                           nthread=4, 
                           seed=27)

clf = GridSearchCV(classifier, param_grid, scoring='roc_auc', cv =5, verbose=True)

l_best_params = []
for i in range(len(l_x_full_train)): 
    x_full_train = l_x_full_train[i]
    clf.fit(x_full_train, y_full_train)
    l_best_params.append(clf.best_params_)
    print(clf.best_params_)
    print(clf.cv_results_['mean_test_score'])
    print(clf.cv_results_['std_test_score'])

# After all optimal parameters are identified for each dataset,
# apply them in the syntax (see below) for model learning
'''


### XGBoost ###
# The following part trains the xgboost models based on the 100% training data
# and uses cross-validation (CV) to evaluate model performances

# Specify the optimal parameters identified in the previous section for each dataset
xgb_param_v1 = {'learning_rate':0.05,
                'n_estimators':190,
                'max_depth':7,
                'min_child_weight':7, 
                'gamma':6, 
                'subsample':0.7, 
                'colsample_bytree':0.4,
                'objective':'binary:logistic', 
                'nthread':4, 
                'seed':27
                }

xgb_param_v1_1 = {'learning_rate':0.03,
                  'n_estimators':300,
                  'max_depth':13,
                  'min_child_weight':7, 
                  'gamma':6, 
                  'subsample':0.7, 
                  'colsample_bytree':0.4,
                  'objective':'binary:logistic', 
                  'nthread':4, 
                  'seed':27
                  }

xgb_param_v1_2 = {'learning_rate':0.03,
                  'n_estimators':300,
                  'max_depth':15,
                  'min_child_weight':7, 
                  'gamma':6, 
                  'subsample':0.7, 
                  'colsample_bytree':0.4,
                  'objective':'binary:logistic', 
                  'nthread':4, 
                  'seed':27
                  }

xgb_param_v2 = {'learning_rate':0.03,
                'n_estimators':300,
                'max_depth':11,
                'min_child_weight':7, 
                'gamma':6, 
                'subsample':0.7, 
                'colsample_bytree':0.4,
                'objective':'binary:logistic', 
                'nthread':4, 
                'seed':27
                }

xgb_param_tuples = {'learning_rate':0.03,
                    'n_estimators':250,
                    'max_depth':7,
                    'min_child_weight':7, 
                    'gamma':6, 
                    'subsample':0.7, 
                    'colsample_bytree':0.4,
                    'objective':'binary:logistic', 
                    'nthread':4, 
                    'seed':27
                    }

xgb_param_triples = {'learning_rate':0.05,
                     'n_estimators':190,
                     'max_depth':7,
                     'min_child_weight':7, 
                     'gamma':6, 
                     'subsample':0.7, 
                     'colsample_bytree':0.4,
                     'objective':'binary:logistic', 
                     'nthread':4, 
                     'seed':27
                     }

l_xgb_param = [xgb_param_v1, xgb_param_v1_1, xgb_param_v1_2, xgb_param_v2, xgb_param_tuples, xgb_param_triples]
                               
# Split the 100% training data in 10 folds to do CV
# Do this manually so I can save the predicted scores from CVs for ensemble
# (this is necessary when the super learner of the ensemble model is based on another ML model,
# which we used for our other top solutions)

# Specify the number of splits for CV (10-fold)
splitter = model_selection.KFold(n_splits=10, shuffle=True, random_state=0)

ll_x_full_val_xgb_pred, ll_x_full_val_xgb_results =[], []
for i in range(len(l_x_full_train)): 
    l_x_full_val_xgb_results = []
    l_x_full_val_xgb_pred = np.zeros(y_full_train.shape)
    x_full_train = l_x_full_train[i]
    xgb_param = l_xgb_param[i]
    for j, (fit_index, val_index) in enumerate(splitter.split(x_full_train, y_full_train)):
        x_full_fit = x_full_train.iloc[fit_index,:]
        y_full_fit = y_full_train.iloc[fit_index]
        x_full_val = x_full_train.iloc[val_index,:]
        y_full_val = y_full_train.iloc[val_index]
        xgb_classifier = XGBClassifier(**xgb_param)
        xgb_classifier.fit(x_full_fit, y_full_fit)
        this_pred = xgb_classifier.predict_proba(x_full_val)[:,1].tolist()
        l_x_full_val_xgb_pred[val_index] = this_pred
        l_x_full_val_xgb_results.append(roc_auc_score(y_full_val, xgb_classifier.predict_proba(x_full_val)[:,1]))
    ll_x_full_val_xgb_pred.append(l_x_full_val_xgb_pred)
    ll_x_full_val_xgb_results.append(l_x_full_val_xgb_results)

print(np.mean(ll_x_full_val_xgb_results, axis=1))
print(np.std(ll_x_full_val_xgb_results, axis=1))

'''
Results:
X_clean_v1, X_clean_v1_1, X_clean_v1_2, X_clean_v2, X_clean_tuples, X_clean_triples
AUC mean:[ 0.82460073  0.82653982  0.82555899  0.82579163  0.82582537  0.82409304]
AUC sd:  [ 0.00894878  0.00839077  0.00938683  0.0087846   0.00880287  0.00778099]
'''

# Ensemble xgb models by average; test performance against y_full_train
ll_x_full_val_xgb_pred_average = pd.DataFrame(ll_x_full_val_xgb_pred).T.mean(axis=1)
print('roc_auc = ', roc_auc_score(y_full_train, ll_x_full_val_xgb_pred_average))

'''
roc_auc =  0.827084937507
'''


### FINAL SUBMISSION ###

# Now that the base classifiers are trained and tested via CV,
# I'm going to use them to generate predicted scores for the 
# TRUE testing data for submission

# Fit the base classifiers on the 100% training data
# and use the fitted models to generate predicted scores from the TRUE testing data (X)

# At the same time, I'm asking for feature importance to see
# which predictors are weighted the most

l_x_true_xgb_pred = []
l_xgb_feature_importance = []
for i in range(len(l_x_true_test)): 
    x_full_train = l_x_full_train[i]
    x_full_test = l_x_true_test[i]
    xgb_param = l_xgb_param[i]
    xgb_classifier = XGBClassifier(**xgb_param)
    xgb_classifier.fit(x_full_train, y_full_train)
    l_x_true_xgb_pred.append(xgb_classifier.predict_proba(x_full_test)[:,1]) # Save predicted scores for ensemble
    # feature importance
    xgb_feature_importance = pd.DataFrame(xgb_classifier.feature_importances_, columns=["weight"], index=x_full_train.columns)
    # xgb_feature_importance.index = x_full_train.columns
    xgb_feature_importance = xgb_feature_importance.sort_values(['weight'], ascending=False)
    l_xgb_feature_importance.append(xgb_feature_importance)

# Compute average predicted scores from individual models based on the TRUE testing data
l_x_true_xgb_pred_average = pd.DataFrame(l_x_true_xgb_pred).T.mean(axis=1)

# Save average predicted scores for final submission
l_x_true_xgb_pred_average_out = pd.concat([x_true_test_Global_ID, pd.DataFrame(l_x_true_xgb_pred_average, columns=['Y_ExitProbability'])], axis = 1)
l_x_true_xgb_pred_average_out.to_csv('C:/Users/Meliu/Desktop/ML Competition/01. SIOP Prep/xgb_pred_average.csv', index=False)

