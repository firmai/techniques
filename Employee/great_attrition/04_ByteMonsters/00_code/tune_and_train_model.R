# parameter grid search sort of...
best_param = list()
best_seednumber = 1234
best_logloss = 0
best_logloss_index = 0


#str(mdcv)
#mdcv$callbacks

#why random vs grid search?
# cause i found the code for the random first, lol 
# 2 https://medium.com/rants-on-machine-learning/smarter-parameter-sweeps-or-why-grid-search-is-plain-stupid-c17d97a0e881
for (iter in 1:250) {
  print(iter)
  param <- list(objective = "binary:logistic",
                eval_metric = "auc",
                #num_class = 12,
                max_depth = sample(6:10, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.2),
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8),
                min_child_weight = sample(1:40, 1)
                )
  cv.nround = 1000
  cv.nfold = 8
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data    = dtrain,
                 label   = train.label, params = param, nthread=7,
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = T, early_stopping_rounds=10)

  #min_logloss = min(mdcv[, test.mlogloss.mean]) data structure has changed, also we want auc
  #str(mdcv)
  min_logloss = max(mdcv$evaluation_log[,test_auc_mean]) # change the min to the max, as auc is not a golf score
  min_logloss_index = which.max(mdcv$evaluation_log[,test_auc_mean])

  if (min_logloss > best_logloss) {
    best_logloss = min_logloss
    best_logloss_index = min_logloss_index
    best_seednumber = seed.number
    best_param = param
  }
}
str(best_param)

nround = best_logloss_index
set.seed(best_seednumber)

save(best_param, file="00_data/sub_yours/best_param.RData")
save(best_logloss_index, file="00_data/sub_yours/best_logloss_index.RData")
save(best_seednumber, file="00_data/sub_yours/best_seednumber.RData")
save(best_logloss, file="00_data/sub_yours/best_logloss.RData")
