library(plyr)
library(gmodels)
library(mice)

# open raw dataset
startup<- read.csv(file="final_data.csv", header=TRUE,as.is=T)

# replace by NA
startup[startup=="No Info"]<- NA
startup[startup==""]<- NA

# create tmp df for columnwise analysis
new.df1 <- data.frame()
new.df1 <- startup

# delete columns with NA > 10%
i <- 1
while(TRUE){
  if(i >= (ncol(new.df1)+1))
    break
  tmp3 <- sum(is.na(new.df1[,i]))
  if((tmp3)>(nrow(new.df1)/10)){
    new.df1 <- new.df1[-i]
    i <- i-1
  }
  i <- i+1
}  

# create tmp df for rowwise analysis
new.df2 <- data.frame()
new.df2 <- new.df1

# delete rows with NA > 10%
i <- 1
while(TRUE){
  if(i >= (nrow(new.df2)+1))
    break
  tmp3 <- sum(is.na(new.df2[i,]))
  if((tmp3)>(ncol(new.df2)/10)){
    new.df2 <- new.df2[-i,]
    i <- i-1
  }
  i <- i+1
}  

# uppercase dataset
new.df2 <- data.frame(lapply(new.df2, function(v) {
  if (is.character(v)) return(toupper(v))
  else return(v)
}))

# new.df2 is the df after rowwise and columnwise missing value handling
write.csv(new.df2,"new.df2.csv")

# imputed_Data <- mice(new_startup, m=5,defaultMethod = c("pmm","logreg", "polyreg", "polr"), maxit = 10, seed = NA)

# imputation for continous attributes
continuos_col <- c(4:7)-1
subset1 <- new.df2[continuos_col]
colnames(subset1)
summary(subset1)

imputed_Data_Continuous <- mice(subset1, m=5,method="pmm", maxit = 10, seed = NA)
subset1 <- complete(imputed_Data_Continuous,5)


# imputation for multivariate categorial attributes
mul_cat_col <- c(8,10,13,14,17,18)-1
subset2 <- new.df2[mul_cat_col]
colnames(subset2)
summary(subset2)

imputed_Data_mul_cat <- mice(subset2, m=5,method="polyreg", maxit = 10, seed = NA)
subset2 <- complete(imputed_Data_mul_cat,5)

# imputation for bivariate categorial attributes
bi_cat_col <- c(9,11:12,15:16,19:23)-1
subset3 <- new.df2[bi_cat_col]
colnames(subset3)
summary(subset3)

imputed_Data_bi_cat <- mice(subset3, m=5,method="logreg", maxit = 10, seed = NA)
subset3 <- complete(imputed_Data_bi_cat,5)

subset4 <- new.df2[1:2]

new.df3 <- data.frame(subset4,subset1, subset2, subset3)

# dataframe with no NA 
write.csv(new.df3,"new.df3.csv")

for(i in 1:ncol(new.df3)){
  if(sum(is.na(new.df3[i]))>0)
    print(i)
}

is.na(new.df3[3])


