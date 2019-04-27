#Install pacman package if you don't have it to help manage installing and loading other packages
  #install.packages("pacman")
  #install lateset version of Xgboost
  #install.packages("drat", repos="https://cran.rstudio.com")
  #drat:::addRepo("dmlc")
  #install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
pacman::p_load(tidyr,dplyr,caret,xgboost)
options(na.action='na.pass') #Set gloabal option to pass through missing values in tree functions called later

set.seed(1234) #Set a seed so results are reproducible

#Reads in training and test set
pretrain <- read.csv(file = "00_data/TrainingSet.csv", stringsAsFactors = FALSE,na.strings = "")
pretest<- read.csv(file = "00_data/TestSet.csv", stringsAsFactors = FALSE,na.strings = "")

source('00_code/external_turnover_data_and_test_features.R')
#Separate code file that engineers some new features:
#1) Calculate observed turnover rates as a function of
#    a) Country
#    b) Job type
#    c) Job function
#2) Unemploymenet rates in each country

#Make sure the test and train set have all the same variables
setdiff(names(pretrain), names(pretest)) #identifies differences
#Put NA in columns that were missing from the training set
pretest$Y_Exit <- NA
pretest$Y_ExitDate <-NA 
pretest$Y_ExitReason <-NA 

mydata<-rbind(pretrain, pretest) #bind test and train together


##############
# build some features 
source("00_code/feature_engineering.R")


#Split test and train ID's out for use later
train_GlobalID<-mydata[!is.na(mydata$Y_Exit),"Global_ID"]
test_GlobalID<-mydata[is.na(mydata$Y_Exit),"Global_ID"]

#drops some uneeded variables like ID that we don't want to one-hot encode later
mydata <- subset(mydata, select = c(Y_Exit, X_TenureDays2009:X_GLDP,S_TenureDays2009:RaceEthnicity_Sim_Count,S_OverallPerformanceRating_mean:X_PMB_Values__sd))

#Turns character data into factors
mydata <- as.data.frame(unclass(mydata))

#Turns NA's into 'Unknown' for factors 
mydata <- purrr::map_df(mydata, ~ {
  if (is.factor(.)) {
    levels(.) <- c(levels(.), 'Unknown')
    .[is.na(.)] <- 'Unknown'
  }
  .
})

#Grabs the train data only
train <- mydata[!is.na(mydata$Y_Exit),]
train.label <- train$Y_Exit

#####Creates Test and train data using TRUE test and train data
#Uncomment if generating actual predictions for submission
test <- mydata[ is.na(mydata$Y_Exit),]

#Not sure what these next two lines are for. Global idea got putlled out a few lines above so we didn't one-hot-encode it. We probably need to delete these next two lines.
names(test)
head(test$Global_ID) == head(test$Global_ID1)

#####Creates splits train data into test and train for model building
#Comment out if generating actual predictions for submission
# splitIndex <- createDataPartition(train$Y_Exit, p = .85, list = FALSE, times = 1)
# train <- train[ splitIndex,]
# test <- train[-splitIndex,]
##############################################

#Create Criterion labels for Xgboost
train.label <- train$Y_Exit
test.label<- test$Y_Exit

print(table(train$Y_Exit))#counts of stayers vs. leavers
print(table(test$Y_Exit)) #Will be empty for true test set
########################

#Imput missing scores for numeric variables
#Need to do separately for test and train data so as to not encode info from test set in train set
pacman::p_load(Hmisc)
nums <- sapply(test, is.numeric) #flags numeric variables
nums[1]<- FALSE #drop target from the list
test[,nums] <- sapply(test[,nums], impute) #impute median for each column
#Standardize Numeric variables except for target
test[,nums] <- scale(test[,nums])
#Drop any columns that still have NA's; shouldn't be any; just being safe
predictors <- test[-1] %>% #Grabs columns except for the target
  select_if(~ !any(is.na(.))) #Drops the ones with NA's (note the entire columns is NA)

#Puts Target and predictors back together
Y_Exit<-test$Y_Exit
test<-cbind(Y_Exit,predictors)
#Repeat on train data
nums <- sapply(train, is.numeric) #flags numeric variables
nums[1]<- FALSE #drop target from the list
train[,nums] <- sapply(train[,nums], impute) #impute median for each column
#Standardize Numeric variables except for target
train[,nums] <- scale(train[,nums])
#Drop any columns that still have NA's; shouldn't be any; just being safe
predictors <- train[-1] %>% #Grabs columns except for the target
  select_if(~ !any(is.na(.))) #Drops the ones with NA's (note the entire columns is NA)
#Puts Target and predictors back together
Y_Exit<-train$Y_Exit
train<-cbind(Y_Exit,predictors)

names(train) %>% sort

# Load the Matrix package
pacman::p_load(Matrix)
# Create sparse matrixes and perform One-Hot Encoding to create dummy variables
dtrain  <- sparse.model.matrix(Y_Exit ~ .-1, data=train)
dtest   <- sparse.model.matrix(Y_Exit ~ .-1, data=test)

# View the number of rows and features of each set
dim(dtrain)
dim(dtest)

