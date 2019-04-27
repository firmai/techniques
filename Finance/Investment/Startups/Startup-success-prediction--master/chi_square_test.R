library(plyr)
library(gmodels)

# input file
startup <- read.csv(file="new.df3.csv", header=TRUE,as.is=T)

# remove 1st col
startup <- startup[,-1]

# init temp df
chi.square.df <- data.frame()

# find chi square value for each attr
for(i in 7:ncol(startup)){
  output <- chisq.test(startup[[2]],startup[[i]])
  print(paste(colnames(startup[i]),output$p.value))
  chi.square.df[i-6,1] <- colnames(startup[i])
  chi.square.df[i-6,2] <- output$p.value
}

# add columnnames to chi.square dataframe
colnames(chi.square.df) <- c("Attribute","Chi_square_value") 

# find attributes with significance level <= 0.05
cnt3 <- 0
for(i in 1:nrow(chi.square.df)){
  val <- chi.square.df[i,2]
  if(val<0.05){
    cnt3 <- cnt3 + 1
    print(paste(i,val))
  }
}

# feature selection is DONE
startup <- startup[(c(1:6,c(1:7,10,15:16)+6))]

# write final data frame
write.csv(startup,"final_clean_data.csv")
