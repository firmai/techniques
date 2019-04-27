# input file
startup <- read.csv(file="after_annova.csv", header=TRUE,as.is=T)

# remove 1st col
startup <- startup[,-c(1,2,5)]
# init temp df
chi.square.df <- data.frame()

# find chi square value for each attr
for(i in 3:ncol(startup)){
  output <- chisq.test(startup[[1]],startup[[i]])
  print(paste(colnames(startup[i]),output$p.value))
  chi.square.df[i-2,1] <- colnames(startup[i])
  chi.square.df[i-2,2] <- output$p.value
  chi.square.df[i-2,3] <- i
}

# add columnnames to chi.square dataframe
colnames(chi.square.df) <- c("Attribute","Chi_square_value","Column_Number") 


ordered_chi_square <- chi.square.df[order(chi.square.df$Chi_square_value),]

tmp_df <- ordered_chi_square[1,]
tmp_df <- tmp_df[-1,]
# find attributes with significance level <= 0.01
cnt <- 1
for(i in 1:nrow(ordered_chi_square)){
  val <- ordered_chi_square[i,2]
  if(val<0.01){
    tmp_df[cnt,] <- ordered_chi_square[i,]
    cnt <- cnt + 1
  }
}



for(i in 1:nrow(startup)){
  if(identical(startup[i,1],"SUCCESS")==T){
    startup[i,1] = 1
  }else{
    startup[i,1] = 0
  }
  
}

startup[,1] <- as.numeric(startup[,1])
#startup <- startup[,-8]



testing_df <- startup[,c(1:2)]
result <- data.frame()
train <- data.frame()
test <- data.frame()

library(e1071)

for(i in 1:nrow(tmp_df)){
  
  testing_df <- data.frame(testing_df,as.data.frame(startup[,tmp_df[i,3]]))
  
  train <- testing_df[unlist(train_list),]
  test <- testing_df[unlist(test_list),]
 
  #model <- glm(Dependent.Company.Status ~.,family=binomial(link='logit'),data=train)
  #summary(model)
  
  #fitted.results <- predict(model,newdata=subset(test,select=c(2:ncol(test))),type='response')
  #fitted.results <- ifelse(fitted.results > 0.5,1,0)
  
  #misClasificError <- mean(fitted.results != test$Dependent.Company.Status)
  #print(paste('Accuracy',1-misClasificError))
  #result[i,1] <- i
  #result[i,2] <- (1-misClasificError)
  
  
  tuned_parameters <- tune.svm(Dependent.Company.Status~., data = train, gamma = 10^(-5:-1), cost = 10^(-3:1))
  summary(tuned_parameters )
  model_svm <- tuned_parameters$best.model
  #model_svm <- svm(Dependent.Company.Status ~., data = train,  kernel = "radial", gamma = 0.1, cost = 1)
  #model_svm
  my_prediction <- predict(model_svm, newdata=subset(test,select=c(2:ncol(test))))
  my_prediction <- ifelse(my_prediction > 0.5,1,0)
  #my_prediction
  misClasificError <- mean(my_prediction != test$Dependent.Company.Status)
  print(paste('Accuracy',1-misClasificError))
  result[i,1] <- i
  result[i,2] <- (1-misClasificError)
  
   
}

colnames(result) <- c("Iteration","Accuracy")

write.csv(result,"after_forward_feature_selection_svm2.csv")


