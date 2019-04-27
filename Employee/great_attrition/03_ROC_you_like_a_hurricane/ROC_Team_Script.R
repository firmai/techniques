#Load packages that will be used during this script.
library(caret)
library(pROC)
library(car)

#Set the working directory where files are located.
setwd("C:/...")

#Load the training set, the test set, and the data codebook. These have been transformed
#from Excel files to csv files for ease in reading into R.
traindat <- read.csv("TrainingSet.csv",as.is=TRUE)
testdat_real <- read.csv("TestSet.csv",as.is=TRUE)
codebook <- read.csv("DataCodebook.csv",as.is=TRUE)


#Create a performance composite of the 2009 performance dimension ratings

PerfComposite2009 <- rowMeans(traindat[,c("X_PMB_Engagement_2009","X_PMB_Teamwork_2009","X_PMB_Accountability_2009","X_PMB_Action_2009","X_PMB_Values_2009")],na.rm=TRUE)

#Recode JobType, JobSubFunction, and Country from 2009. The purpose of these recodes is to
#collapse low n categories. There is definitely a more compact way to do this; it was just easy
#to program it this way from the Excel sheet where we determined how to collapse the categories.

JobTypeRecode <- recode(traindat[,"X_JobType2009"],"'PATHOLOGIST'='Missing';''='Missing'")

JobSubRecode <- recode(traindat[,"X_JobSubFunction2009"],"''='Missing';
'ACCOUNTING/REPORTING'='ACCOUNTING/REPORTING';
'AFFILIATE LEAD POSITION - PHARMA'='AFFILIATE LEAD POSITION - PHARMA';
'ANIMAL HEALTH RES/DEV'='ANIMAL HEALTH RES/DEV';
'BULK MANUFACTURING OPERATIONS'='BULK MANUFACTURING OPERATIONS';
'CLINICAL OPERATIONS'='CLINICAL OPERATIONS';
'COMMUNICATIONS/COMMUNITY RELATIONS'='COMMUNICATIONS/COMMUNITY RELATIONS';
'COMPENSATION/BENEFITS/RELOCATION'='COMPENSATION/BENEFITS/RELOCATION';
'CORPORATE AFFAIRS EXEC/ADMIN SUPPORT'='CORPORATE AFFAIRS EXEC/ADMIN SUPPORT';
'CORPORATE FUNDED MEDICAL'='CORPORATE FUNDED MEDICAL';
'CORPORATE GENERAL MANAGEMENT'='CORPORATE GENERAL MANAGEMENT';
'DEVICE MANUFACTURING'='DEVICE MANUFACTURING';
'DISCOVERY RESEARCH/RESEARCH TECHNOLOGIES'='DISCOVERY RESEARCH/RESEARCH TECHNOLOGIES';
'DISTRIBUTION'='DISTRIBUTION';
'ENGINEERING'='ENGINEERING';
'ENVIRONMENTAL/HEALTH/SAFETY'='ENVIRONMENTAL/HEALTH/SAFETY';
'EXECUTIVE MANAGEMENT ADMIN SUPPORT'='EXECUTIVE MANAGEMENT ADMIN SUPPORT';
'FACILITY/CORPORATE SECURITY/ERT'='FACILITY/CORPORATE SECURITY/ERT';
'FILL/FINISH MANUFACTURING OPERATIONS'='FILL/FINISH MANUFACTURING OPERATIONS';
'FIN EXEC/ADMIN SUPPORT'='FIN EXEC/ADMIN SUPPORT';
'FINANCIAL OPERATIONS/SERVICES'='FINANCIAL OPERATIONS/SERVICES';
'GENERAL LAW'='GENERAL LAW';
'GLOBAL COMPLIANCE AND ETHICS'='GLOBAL COMPLIANCE AND ETHICS';
'GLOBAL HEALTH OUTCOMES'='GLOBAL HEALTH OUTCOMES';
'GOVERNMENT/PUBLIC/ADVOCACY'='GOVERNMENT/PUBLIC/ADVOCACY';
'HR EXEC ADMIN SUPPORT/BUSINESS STAFF'='HR EXEC ADMIN SUPPORT/BUSINESS STAFF';
'IT BUSINESS INTEGRATION/AFF'='IT BUSINESS INTEGRATION/AFF';
'LEGAL EXEC/ADMIN SUPPORT/BUSINESS STAFF'='LEGAL EXEC/ADMIN SUPPORT/BUSINESS STAFF';
'LINE HR'='LINE HR';
'MAINTENANCE'='MAINTENANCE';
'MANUFACTURING TECHNICAL SERVICES'='MANUFACTURING TECHNICAL SERVICES';
'MARKETING-PHARMA'='MARKETING-PHARMA';
'MFG STRATEGY/EXEC/ADMIN SUPPORT'='MFG STRATEGY/EXEC/ADMIN SUPPORT';
'MKT/SALES EXEC/ADMIN SUPPORT-PHARMA'='MKT/SALES EXEC/ADMIN SUPPORT-PHARMA';
'OFFICE SERVICES'='OFFICE SERVICES';
'PACKAGING'='PACKAGING';
'PATENT LAW'='PATENT LAW';
'PLANNING/CONTROLLER'='PLANNING/CONTROLLER';
'PROCUREMENT'='PROCUREMENT';
'PRODUCT/PROCESS DEVELOPMENT'='PRODUCT/PROCESS DEVELOPMENT';
'PROJECT MANAGEMENT'='PROJECT MANAGEMENT';
'QUALITY ASSURANCE'='QUALITY ASSURANCE';
'QUALITY CONTROL'='QUALITY CONTROL';
'RECRUITING/STAFFING'='RECRUITING/STAFFING';
'REGULATORY'='REGULATORY';
'SALES - PHARMA'='SALES - PHARMA';
'SALES/MARKETING SUPPORT'='SALES/MARKETING SUPPORT';
'SALES/MKTG-ANIMAL HEALTH'='SALES/MKTG-ANIMAL HEALTH';
'SCIENCE/TECHNOLOGY GEN ADMIN/ADMIN SUPPT'='SCIENCE/TECHNOLOGY GEN ADMIN/ADMIN SUPPT';
'SIX SIGMA-MANUFACTURING'='SIX SIGMA-MANUFACTURING';
'SIX SIGMA-SALES/MARKETING'='SIX SIGMA-SALES/MARKETING';
'SIX SIGMA-SCIENCE AND TECH'='SIX SIGMA-SCIENCE AND TECH';
'STRATEGY/BUS DEV/ASSETS/CFIB'='STRATEGY/BUS DEV/ASSETS/CFIB';
'SUPPLY CHAIN/MATERIALS MANAGEMENT'='SUPPLY CHAIN/MATERIALS MANAGEMENT';
'TOXICOLOGY/ADME'='TOXICOLOGY/ADME';
'TRAINING/DEVELOPMENT'='TRAINING/DEVELOPMENT';
else='Missing'")

CountryRecode <- recode(traindat[,"X_Country2009"],"''='Missing';
'Argentina'='Argentina';
'Australia'='Australia';
'Austria'='Austria';
'Belgium'='Belgium';
'Brazil'='Brazil';
'Canada'='Canada';
'China'='China';
'Columbia'='Columbia';
'Czech Rebublic'='Czech Rebublic';
'Denmark'='Denmark';
'Egypt'='Egypt';
'Finlan'='Finlan';
'France'='France';
'Germany'='Germany';
'Greece'='Greece';
'Hong Kong'='Hong Kong';
'Hungary'='Hungary';
'India'='India';
'Ireland'='Ireland';
'Israel'='Israel';
'Italy'='Italy';
'Japan'='Japan';
'Lebanon'='Lebanon';
'Malaysia'='Malaysia';
'Mexico'='Mexico';
'Netherlands'='Netherlands';
'Norway'='Norway';
'Pakistan'='Pakistan';
'Peru'='Peru';
'Philippines'='Philippines';
'Poland'='Poland';
'Portugal'='Portugal';
'Puerto Rico'='Puerto Rico';
'Romania'='Romania';
'Russia'='Russia';
'Saudia Arabia'='Saudia Arabia';
'Singapore'='Singapore';
'South Africa'='South Africa';
'South Korea'='South Korea';
'Spain'='Spain';
'Sweeden'='Sweeden';
'Switzerland'='Switzerland';
'Taiwan'='Taiwan';
'Thailand'='Thailand';
'Turkey'='Turkey';
'United Kingdom'='United Kingdom';
'United States of America'='United States of America';
'Venezuela'='Venezuela';
else='Missing'")


#Add the new variables to the training dataset.
traindat <- cbind(traindat,PerfComposite2009,JobSubRecode,CountryRecode,JobTypeRecode)

#Create a variable with the full list of predictors; useful for running models involving
#all the variables.
preds <- names(traindat)[6:ncol(traindat)]

#List of variables included in the parsimony model that produced the highest cross-validated
#AUC for us.

preds2 <- c("CountryRecode","JobSubRecode","X_TenureDays2009","X_OverallPerformanceRating2009",
"X_Age2009","JobTypeRecode","PerfComposite2009")

#Convert criterion to a factor to ensure it is analyzed using classification models
#by caret.

traindat$Crit <- ifelse(traindat[,"Y_Exit"]==1,'Left','Stayed')
traindat$Crit <- as.factor(traindat[,"Crit"])

#Convert string variable predictors to factors. This uses the VariableType provided
#in the data codebook to identify which variables are "String" variables.

stringvars <- subset(codebook,VariableType=="String")
stringnames <- stringvars[,"VariableName"]
traindat[,stringnames] <- lapply(traindat[,stringnames],factor)

#Split training sample into two parts (test and training) to evaluate models.
#Note that a seed is set to replicate this split.

set.seed(5)
datasplit <- sample(c(1,0),nrow(traindat),replace=TRUE,prob=c(.75,.25))

traindat_train <- traindat[which(datasplit==1),]
traindat_test <- traindat[which(datasplit==0),]

#Split test set (subset of training set) into 2 to mimic the public/private leaderboard.
datasplit_test <- sample(c(1,0),nrow(traindat_test),replace=TRUE,prob=c(.50,.50))

traindat_test1 <- traindat_test[which(datasplit_test==0),]
traindat_test2 <- traindat_test[which(datasplit_test==1),]

#Run gbm model with specific tuning parameter ranges that produced our highest cross-validated AUC.

gbm_grid <- expand.grid(interaction.depth=c(1,2,3,6,9),n.trees=(1:10)*50,shrinkage=c(.01,.005,.05,.1),n.minobsinnode=c(5,10,15))

gbm_mod <- train(traindat_train[,preds2],traindat_train[,"Crit"],
        method = "gbm",
        trControl=trainControl(method='cv', number=10, repeats=3, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE,savePredictions='final'),
        metric="ROC",tuneGrid=gbm_grid,
        preProc=c("center","scale"))

#Apply the model to the first test set of the training data to obtain predicted probabilities.

gbm_predictions1 <- predict(object=gbm_mod, traindat_test1[,preds2], type='prob')

#Calculate the auc for the first test set
auc <- roc(traindat_test1[,"Crit"],gbm_predictions1[,1])

auc$auc

#Apply the model to the second test set of the training data to obtain predicted probabilities.

gbm_predictions2 <- predict(object=gbm_mod, traindat_test2[,preds2], type='prob')

#Calculate the auc for the second test set
auc <- roc(traindat_test2[,"Crit"],gbm_predictions2[,1])

auc$auc

#After evaluating the AUCs and deciding that this model was a good one to pursue,
#run model on the full training dataset so all the training data is used to train the model.

gbm_grid <- expand.grid(interaction.depth=c(1,2,3,6,9),n.trees=(1:10)*50,shrinkage=c(.01,.005,.05,.1),n.minobsinnode=c(5,10,15))

gbm_mod_full <- train(traindat[,preds2],traindat[,"Crit"],
        method = "gbm",
        trControl=trainControl(method='cv', number=10, repeats=3, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE,savePredictions='final'),
        metric="ROC",tuneGrid=gbm_grid,
        preProc=c("center","scale"))


#Prepare the test dataset in the same manner as the training dataset was modified.
#This will allow the model object to be applied to the test dataset to create
#predicted probabilities.

PerfComposite2009Test <- rowMeans(testdat_real[,c("X_PMB_Engagement_2009","X_PMB_Teamwork_2009","X_PMB_Accountability_2009","X_PMB_Action_2009","X_PMB_Values_2009")],na.rm=TRUE)

JobTypeRecodeTest <- recode(testdat_real[,"X_JobType2009"],"'PATHOLOGIST'='Missing';''='Missing'")

JobSubRecodeTest <- recode(testdat_real[,"X_JobSubFunction2009"],"''='Missing';
'ACCOUNTING/REPORTING'='ACCOUNTING/REPORTING';
'AFFILIATE LEAD POSITION - PHARMA'='AFFILIATE LEAD POSITION - PHARMA';
'ANIMAL HEALTH RES/DEV'='ANIMAL HEALTH RES/DEV';
'BULK MANUFACTURING OPERATIONS'='BULK MANUFACTURING OPERATIONS';
'CLINICAL OPERATIONS'='CLINICAL OPERATIONS';
'COMMUNICATIONS/COMMUNITY RELATIONS'='COMMUNICATIONS/COMMUNITY RELATIONS';
'COMPENSATION/BENEFITS/RELOCATION'='COMPENSATION/BENEFITS/RELOCATION';
'CORPORATE AFFAIRS EXEC/ADMIN SUPPORT'='CORPORATE AFFAIRS EXEC/ADMIN SUPPORT';
'CORPORATE FUNDED MEDICAL'='CORPORATE FUNDED MEDICAL';
'CORPORATE GENERAL MANAGEMENT'='CORPORATE GENERAL MANAGEMENT';
'DEVICE MANUFACTURING'='DEVICE MANUFACTURING';
'DISCOVERY RESEARCH/RESEARCH TECHNOLOGIES'='DISCOVERY RESEARCH/RESEARCH TECHNOLOGIES';
'DISTRIBUTION'='DISTRIBUTION';
'ENGINEERING'='ENGINEERING';
'ENVIRONMENTAL/HEALTH/SAFETY'='ENVIRONMENTAL/HEALTH/SAFETY';
'EXECUTIVE MANAGEMENT ADMIN SUPPORT'='EXECUTIVE MANAGEMENT ADMIN SUPPORT';
'FACILITY/CORPORATE SECURITY/ERT'='FACILITY/CORPORATE SECURITY/ERT';
'FILL/FINISH MANUFACTURING OPERATIONS'='FILL/FINISH MANUFACTURING OPERATIONS';
'FIN EXEC/ADMIN SUPPORT'='FIN EXEC/ADMIN SUPPORT';
'FINANCIAL OPERATIONS/SERVICES'='FINANCIAL OPERATIONS/SERVICES';
'GENERAL LAW'='GENERAL LAW';
'GLOBAL COMPLIANCE AND ETHICS'='GLOBAL COMPLIANCE AND ETHICS';
'GLOBAL HEALTH OUTCOMES'='GLOBAL HEALTH OUTCOMES';
'GOVERNMENT/PUBLIC/ADVOCACY'='GOVERNMENT/PUBLIC/ADVOCACY';
'HR EXEC ADMIN SUPPORT/BUSINESS STAFF'='HR EXEC ADMIN SUPPORT/BUSINESS STAFF';
'IT BUSINESS INTEGRATION/AFF'='IT BUSINESS INTEGRATION/AFF';
'LEGAL EXEC/ADMIN SUPPORT/BUSINESS STAFF'='LEGAL EXEC/ADMIN SUPPORT/BUSINESS STAFF';
'LINE HR'='LINE HR';
'MAINTENANCE'='MAINTENANCE';
'MANUFACTURING TECHNICAL SERVICES'='MANUFACTURING TECHNICAL SERVICES';
'MARKETING-PHARMA'='MARKETING-PHARMA';
'MFG STRATEGY/EXEC/ADMIN SUPPORT'='MFG STRATEGY/EXEC/ADMIN SUPPORT';
'MKT/SALES EXEC/ADMIN SUPPORT-PHARMA'='MKT/SALES EXEC/ADMIN SUPPORT-PHARMA';
'OFFICE SERVICES'='OFFICE SERVICES';
'PACKAGING'='PACKAGING';
'PATENT LAW'='PATENT LAW';
'PLANNING/CONTROLLER'='PLANNING/CONTROLLER';
'PROCUREMENT'='PROCUREMENT';
'PRODUCT/PROCESS DEVELOPMENT'='PRODUCT/PROCESS DEVELOPMENT';
'PROJECT MANAGEMENT'='PROJECT MANAGEMENT';
'QUALITY ASSURANCE'='QUALITY ASSURANCE';
'QUALITY CONTROL'='QUALITY CONTROL';
'RECRUITING/STAFFING'='RECRUITING/STAFFING';
'REGULATORY'='REGULATORY';
'SALES - PHARMA'='SALES - PHARMA';
'SALES/MARKETING SUPPORT'='SALES/MARKETING SUPPORT';
'SALES/MKTG-ANIMAL HEALTH'='SALES/MKTG-ANIMAL HEALTH';
'SCIENCE/TECHNOLOGY GEN ADMIN/ADMIN SUPPT'='SCIENCE/TECHNOLOGY GEN ADMIN/ADMIN SUPPT';
'SIX SIGMA-MANUFACTURING'='SIX SIGMA-MANUFACTURING';
'SIX SIGMA-SALES/MARKETING'='SIX SIGMA-SALES/MARKETING';
'SIX SIGMA-SCIENCE AND TECH'='SIX SIGMA-SCIENCE AND TECH';
'STRATEGY/BUS DEV/ASSETS/CFIB'='STRATEGY/BUS DEV/ASSETS/CFIB';
'SUPPLY CHAIN/MATERIALS MANAGEMENT'='SUPPLY CHAIN/MATERIALS MANAGEMENT';
'TOXICOLOGY/ADME'='TOXICOLOGY/ADME';
'TRAINING/DEVELOPMENT'='TRAINING/DEVELOPMENT';
else='Missing'")

CountryRecodeTest <- recode(testdat_real[,"X_Country2009"],"''='Missing';
'Argentina'='Argentina';
'Australia'='Australia';
'Austria'='Austria';
'Belgium'='Belgium';
'Brazil'='Brazil';
'Canada'='Canada';
'China'='China';
'Columbia'='Columbia';
'Czech Rebublic'='Czech Rebublic';
'Denmark'='Denmark';
'Egypt'='Egypt';
'Finlan'='Finlan';
'France'='France';
'Germany'='Germany';
'Greece'='Greece';
'Hong Kong'='Hong Kong';
'Hungary'='Hungary';
'India'='India';
'Ireland'='Ireland';
'Israel'='Israel';
'Italy'='Italy';
'Japan'='Japan';
'Lebanon'='Lebanon';
'Malaysia'='Malaysia';
'Mexico'='Mexico';
'Netherlands'='Netherlands';
'Norway'='Norway';
'Pakistan'='Pakistan';
'Peru'='Peru';
'Philippines'='Philippines';
'Poland'='Poland';
'Portugal'='Portugal';
'Puerto Rico'='Puerto Rico';
'Romania'='Romania';
'Russia'='Russia';
'Saudia Arabia'='Saudia Arabia';
'Singapore'='Singapore';
'South Africa'='South Africa';
'South Korea'='South Korea';
'Spain'='Spain';
'Sweeden'='Sweeden';
'Switzerland'='Switzerland';
'Taiwan'='Taiwan';
'Thailand'='Thailand';
'Turkey'='Turkey';
'United Kingdom'='United Kingdom';
'United States of America'='United States of America';
'Venezuela'='Venezuela';
else='Missing'")

testdat_real <- cbind(testdat_real,PerfComposite2009Test,JobSubRecodeTest,CountryRecodeTest,JobTypeRecodeTest)

#Rename the variables so that the model is able to be applied to the exact variable names.
colnames(testdat_real)[which(colnames(testdat_real)=="PerfComposite2009Test")] <- "PerfComposite2009"
colnames(testdat_real)[which(colnames(testdat_real)=="JobSubRecodeTest")] <- "JobSubRecode"
colnames(testdat_real)[which(colnames(testdat_real)=="CountryRecodeTest")] <- "CountryRecode"
colnames(testdat_real)[which(colnames(testdat_real)=="JobTypeRecodeTest")] <- "JobTypeRecode"

#Change the test set variables to factors.
stringnames2 <- c("JobSubRecode","CountryRecode","JobTypeRecode")
testdat_real[,stringnames2] <- lapply(testdat_real[,stringnames2],factor)

#Obtain predicted probabilities by applying the model to the test dataset.
gbm_testpreds <- predict(object=gbm_mod_full,testdat_real[,preds2],type='prob')

#Combine predicted probabilities with IDs for test set.
resdat <- data.frame(Global_ID=testdat_real[,"Global_ID"],Y_ExitProbability=gbm_testpreds[,1])

#Write the predicted probabilities to a csv file for submission.
write.csv(resdat,"ROCYouLikeAHurricaneEntryX.csv",row.names=FALSE)




