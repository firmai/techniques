# this code averages the other three to produce the most stable predictions

rm(list=ls(all=TRUE)) #cleaning up workspace
#Using first three models to create our fourth winning entry
#reading results of first three models back into workspace
sub1<- read.csv(file = "ByteMonsters022618_Entry1.csv", stringsAsFactors = FALSE,na.strings = "")
sub2<- read.csv(file = "ByteMonsters022618_Entry2.csv", stringsAsFactors = FALSE,na.strings = "")
sub3<- read.csv(file = "ByteMonsters022618_Entry3.csv", stringsAsFactors = FALSE,na.strings = "")
#Merging three models together
test<- merge(sub1, sub2, by = "Global_ID")
test2<-  merge(test, sub3, by = "Global_ID")
#Averaging the 3 models
test2$means<- rowMeans( test2[,2:4])
test2<-test2 %>% select( Global_ID, Y_ExitProbability =means)
#Writing result to file
write.csv(test2,file="ByteMonsters022618_Entry4.csv", row.names = FALSE)
