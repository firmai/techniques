library(plyr)
library(gmodels)

startup <- read.csv(file="after_chi_sqr_0.01.csv", header=TRUE,as.is=T)

startup <- startup[,-c(1:2)]
startup <- startup[,-3]

startup <- startup[,c(1,2,4,5,11,14,17)]
one_cnt <- 0
zero_cnt <- 0 
for(i in 1:nrow(startup)){
  if(identical(startup[i,1],"SUCCESS")==T){
    startup[i,1] = 1
    one_cnt = one_cnt + 1
  }else{
    startup[i,1] = 0
    zero_cnt = zero_cnt + 1
  }
  
}

startup[,1] <- as.numeric(startup[,1])
#startup <- startup[,-8]

one_cnt
zero_cnt

for(i in 3:ncol(startup)){
  startup[,i] <- as.factor(startup[,i])
}


reqd_cnt1 = 115
reqd_cnt2 = 115

train <- startup[1,]
test <- startup[1,]
train = train[-1,]
test = test[-1,]
train_cnt <- 1
test_cnt <- 1
train_list <- list()
test_list <- list()

for(i in 1:nrow(startup)){
  val = startup[i,1]
  if(val==1){
    if(reqd_cnt1>0){
      train[train_cnt,] = startup[i,]
      train_list[train_cnt] <- i
      train_cnt = train_cnt + 1
      reqd_cnt1 = reqd_cnt1 - 1
    }else{
      test[test_cnt,] = startup[i,]
      test_list[test_cnt] <- i
      test_cnt = test_cnt + 1
    }
  }else{
    if(reqd_cnt2>0){
      train[train_cnt,] = startup[i,]
      train_list[train_cnt] <- i
      train_cnt = train_cnt + 1
      reqd_cnt2 = reqd_cnt2 - 1
    }else{
      test[test_cnt,] = startup[i,]
      test_list[test_cnt] <- i
      test_cnt = test_cnt + 1
    }
  }
}



model <- glm(Dependent.Company.Status ~.,family=binomial(link='logit'),data=train)
summary(model)

fitted.results <- predict(model,newdata=subset(test,select=c(2:ncol(test))),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != test$Dependent.Company.Status)
print(paste('Accuracy',1-misClasificError))


sum(is.na(startup[,3]))

