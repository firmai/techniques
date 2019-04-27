

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

library(rpart)
# grow tree 
fit <- rpart(Dependent.Company.Status ~., data = train,method="class")
 summary(fit)
 fit
#Predict Output 
 predicted= predict(fit,test)

 


 
 
  library(rattle)
  library(rpart.plot)
  library(RColorBrewer)
 fancyRpartPlot(fit)
 
 
 output<-as.data.frame(predicted)
 
 output_pred<-data.frame()
 val = ""
 for(i in 1:217)
 {
   
   if(output[i,2] > output[i,1])
   {
      val = "SUCCESS"
   }
   else{
      val = "FAILED"
   }
    print(val)
    output_pred[i,1] = val
 }

 
#nrow(y_test.df)

y_test.df = as.data.frame(test[,1])

for(i in 1:217)
{
  y_test.df$`test[, 1]` = as.character(y_test.df$`test[, 1]`)
} 


correct_cnt <- 0
for(i in 1:217)
{
  print(paste(y_test.df[i,1]," ", output_pred[i,1]," ",identical(y_test.df[i,1], output_pred[i,1])))
  
  if(identical(y_test.df[i,1], output_pred[i,1])){
    correct_cnt = correct_cnt+1
  }
}

acc = (correct_cnt*100/217)
print(acc)

#typeof(output_pred[1,1])
#print(y_test.df[1,1])



 

#typeof(y_test.df[1,1])
