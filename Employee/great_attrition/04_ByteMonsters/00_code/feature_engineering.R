#recode exit reason into 4 categories----
#Stayers, career reasons, personal reasons, Lilly reasons
mydata[is.na(mydata$Y_ExitReason),"Y_ExitReason"] <- "Stayer"
mydata$Y_ExitReason <- recode(mydata$Y_ExitReason, 
                              "Assigned work" = "Lilly", 
                              "Career options" = "Career",
                              "End of Probationary Period-Employee Decision" = "Lilly",
                              "Further education" = "Career",
                              "Lilly policies" = "Lilly", 
                              "Location" = "Personal", 
                              "Major career change" = "Career",
                              "Pay" = "Lilly",
                              "Resigned for personal reasons" = "Personal", 
                              "Supervision" = "Lilly",
                              "Went into business for self" = "Career",
                              "Work shift" = "Lilly",
                              "Work/Family balance" = "Personal" 
)


#Function to create counts of unique values for variables over time ignoring NA----
count.function<- function(varStem) {  
  years <- c(2004:2009)
  vars<-paste(varStem,years,sep="")
  count_var<-paste(varStem,"count",sep="_") 
  mydata[,count_var]<<-apply(mydata[vars],1,function(x) length(unique(na.omit(x))))
}
#Change in country
count.function("X_Country")
#Change in city
count.function("X_City")
#Change in job function
count.function("X_JobFunction")
#Change in Sub Job function
count.function("X_JobSubFunction")
#Change in Job Type
count.function("X_JobType")
#Change in Job Type
count.function("S_SupervisorGlobal_ID")

#Function to calculate gender and race similarity with supervisor at each time point----
raceGenderSim <- function (type) {
  for (yr in 2004:2009) {
    x<-paste("S_",paste(type,yr,sep=""),sep="") #name of x var
    y<-paste("X_",type,sep="") #name of y var
    resultLabel<-paste(paste(type,"Sim",sep="_"),yr,sep="") #name of result var
    #recode if x & y are equal and not missing
    mydata[,resultLabel]<<-ifelse(mydata[x]==mydata[y], 1, 
                                  ifelse(is.na(mydata[x]),NA,
                                         ifelse(is.na(mydata[y]),NA,0)))
  }
} 
raceGenderSim("Gender") #Computes gender similarity at each yr
raceGenderSim("RaceEthnicity") #Computes race similarity at each yr

#Function to calculate change in supervisor at each time point----
#yr starts in 2005 and compares to year prior
newSuper <- function () {
  for (yr in 2005:2009) {
    x<-paste("S_SupervisorGlobal_ID",(yr-1),sep="")#name of x var
    y<-paste("S_SupervisorGlobal_ID",(yr),sep="") #name of y var
    resultLabel<-paste("S_change_",yr,sep="") #name of result var
    #recode if x & y are equal and not missing
    mydata[,resultLabel]<<-ifelse(mydata[x]==mydata[y], 0, 
                                  ifelse(is.na(mydata[x]),NA,
                                         ifelse(is.na(mydata[y]),NA,1)))
  }
} 
newSuper() #runFunction

#Calculate Age similarity at each time point (simple diff score)----
mydata$Age_Sim04 <- (mydata$X_Age2009-5)-mydata$S_Age2004
mydata$Age_Sim05 <- (mydata$X_Age2009-4)-mydata$S_Age2005
mydata$Age_Sim06 <- (mydata$X_Age2009-3)-mydata$S_Age2006
mydata$Age_Sim07 <- (mydata$X_Age2009-2)-mydata$S_Age2007
mydata$Age_Sim08 <- (mydata$X_Age2009-1)-mydata$S_Age2008
mydata$Age_Sim09 <- (mydata$X_Age2009)-mydata$S_Age2009

#Count number of years with same gender/race supervisor----
mydata$Gender_Sim_Count<-rowSums(subset(mydata, select = c(Gender_Sim2004:Gender_Sim2009)), na.rm=TRUE)
#Count number of years with same race supervisor
mydata$RaceEthnicity_Sim_Count<-rowSums(subset(mydata, select = c(RaceEthnicity_Sim2004:RaceEthnicity_Sim2009)), na.rm=TRUE)

#Summaries of other multi-year numeric statistics----
#Calculates Means, St dev
#Select data, calculated summaries, merges to original
mydata<-subset(mydata, select = c(Global_ID,X_PayGradeLevel2009:X_PMB_ShareKeyLearning_2005,S_PayGradeLevel2009:S_OverallPerformanceRating2004,S_PMB_ModelValues_2008:S_PMB_ShareKeyLearning_2005)) %>%
  gather(key, value, -Global_ID)%>%
  extract(key, c("question", "year"), "(\\D+)(....)")%>%
  spread(question, value)%>%
  select(-year)%>%
  group_by(Global_ID) %>%
  summarise_all(funs(mean(., na.rm = TRUE),sd(., na.rm = TRUE)))%>%
  bind_cols(mydata, .) %>%
  ungroup()

#Clean problematic values----
#Replace INF with NA
mydata <- do.call(data.frame,lapply(mydata, function(x) replace(x, is.infinite(x),NA)))
#Replace NaN with NA
mydata <- do.call(data.frame,lapply(mydata, function(x) replace(x, is.nan(x),NA)))
####################################################
