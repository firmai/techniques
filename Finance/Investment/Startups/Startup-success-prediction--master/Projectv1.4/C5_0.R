
library(C50)

startup <- read.csv("after_chi_sqr_0.01.csv", header = T, stringsAsFactors = F)
startup <-startup[,-c(1,2,5)]

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
  if(identical(val,"SUCCESS")){
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

train[,1] <- as.factor(train[,1])
test[,1] <- as.factor(test[,1])


m1 <- C5.0(train[,2:ncol(train)],train[,1],method="class")
m1
summary(m1)
p1 <-predict(m1,test[,2:ncol(test)])
length(p1)
table(test[1:nrow(test),1],Predicted=p1)

plot(m1)
