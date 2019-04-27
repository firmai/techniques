##########################################
# SIOP 2018 Machine Learning Competition Submission
# Done by Isaac Thompson and Scott Tonidandel
# The data for this was provided by a prominent company and is not released as of yet
# hopefully the code will still be beneficial

# the winning solution was an average of three submissions
# submission one was filled with very specific features
# submission two included more future variables but they were means and broader  
# all the differences in the first two submissions have to do with external_turnover_data_and_test_feature.R
# submission three has none of that sillyness, to reproduce this submission, skip the entire source of external_turnover_data_and_test_feature.R
# submission four was the average of these three. 


#####################################
# creates a nice workplace 
rm(list=ls(all=TRUE)) #clean workspace
setwd("~/00_Git/siop_ml_comp") #set working directory
pacman::p_load(tidyr, dplyr, caret, xgboost)
options(na.action='na.pass') #Set gloabal option to pass through missing values in tree functions called later
set.seed(1234) #Set a seed so results are reproducible

# run this to load the file and seperate training and testing 
source("00_code/data_prep.R")

# run this to find some good parameters 
source("00_code/tune_and_train_model.R")


# load results from best performing grid search 
# just load some good ones that I found if you don't want to find your own
# sub stands for submission, the three submissions were averaged to produce sub 4 which was the winner
# change the path if you want to change the submission parameters 
load("00_data/sub_1/best_param.RData")
load("00_data/sub_1/best_logloss_index.RData")
load("00_data/sub_1/best_seednumber.RData")
load ("00_data/sub_1/best_logloss.RData")

best_param # to look the pamameters you are bringing in
set.seed(best_seednumber)

# train the actual model based on the hyper parameters 
system.time(xgb <- xgboost(params  = best_param,
                           data    = dtrain,
                           label   = train.label,
                           nrounds = best_logloss_index,
                           print_every_n = 10,
                           verbose = 1, 
                           nthread = 7))


pred <- predict(xgb, dtest) # get some predictions
Y_ExitProbability <- pred

pred.resp <- as.numeric(pred >=0.5)
table(pred.resp)


Global_ID<-pretest$Global_ID
result<-cbind(Global_ID,Y_ExitProbability) # end results! Yayyyyay

# # uncomment to write results, name yours what ever you like 
# write.csv(result,file="ByteMonsters022618_Entry1.csv", row.names = FALSE)



# unncomment below and run something like this if you want the winning solution:
# simply an average of the three submissions
# source("00_code/average_three_models.R")
