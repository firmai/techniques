
#rm(list=ls(all=TRUE))

library(lubridate)

#### 
# country, how many left, year
# add variable 
pretrain$year_left<- dmy(pretrain$Y_ExitDate) %>% year %>% as.numeric()

year_left_data <- dplyr::filter(pretrain, !is.na(year_left)) 

how_many_left <- dplyr::filter(pretrain, !is.na(year_left))  %>% 
  group_by(Country =X_Country2009, Year =year_left ) %>%
  summarise(country_turnover =n())

how_many_total <- dplyr::filter(pretrain)  %>% 
  group_by(Country= X_Country2009 ) %>%
  summarise(country_emp_pop=n())

Country_year <- dplyr::filter(pretrain)  %>%
  group_by(Country = X_Country2009 ) %>%
  summarise(n())

pretrain$year_left<-NULL

list1 <- Country_year$Country
list2 <- rep(c( 2009, 2010, 2011, 2012, 2013, 2014), each = length(list1))
Country_year <- as.data.frame(cbind(list1, list2)) 
names(Country_year)<-c("Country", "Year")

Country_year <- merge(Country_year, how_many_total, by="Country", all = TRUE)
Country_year<- merge(Country_year, how_many_left, by = c( "Country", "Year"), all = TRUE)

Country_year[is.na(Country_year)] <- 0
Country_year<- Country_year %>%
  mutate(yearly_turnover_country =country_turnover/country_emp_pop)

#######
Country_year %>% head
# merge back to main data frame... how? 
# create a flatter data set... cast 
library(reshape2)
x<-dcast( Country_year,Country ~ Year, value.var = "yearly_turnover_country" ) 
x<-x %>% setNames(paste0('Country_Turnover_', names(.)))
x<-rename(x, X_Country2009 = Country_Turnover_Country)

nums <- sapply(x, is.numeric)
x$Future_Country_Turnover<-rowMeans(x[ , nums])
head(x)
x <- select ( x, X_Country2009, Future_Country_Turnover)
x$Future_Country_Turnover[x$Future_Country_Turnover==0] <- mean(x$Future_Country_Turnover[!x$Future_Country_Turnover==0])

y<-merge(pretrain, x, all.x = TRUE, by = "X_Country2009") 
pretrain<-y

pretest<-merge(pretest, x, all.x = TRUE, by = "X_Country2009") 
x<-NULL



# job type future turnover 
pretrain$X_JobType2009 %>% unique()

pretrain$year_left<- dmy(pretrain$Y_ExitDate) %>% year %>% as.numeric()

year_left_data <- dplyr::filter(pretrain, !is.na(year_left)) 

how_many_left <- dplyr::filter(pretrain, !is.na(year_left))  %>% 
  group_by(X_JobType2009, Year =year_left ) %>%
  summarise(type_turnover =n())

how_many_total <- dplyr::filter(pretrain)  %>% 
  group_by(X_JobType2009 ) %>%
  summarise(type_emp_pop=n())

type_year <- dplyr::filter(pretrain)  %>%
  group_by(X_JobType2009 ) %>%
  summarise(n())

pretrain$year_left<-NULL

list1 <- type_year$X_JobType2009
list2 <- rep(c(2009, 2010, 2011, 2012, 2013, 2014), each = length(list1))
type_year <- as.data.frame(cbind(list1, list2)) 
names(type_year)<-c("X_JobType2009", "Year")

type_year <- merge(type_year, how_many_total, by="X_JobType2009", all = TRUE)
type_year<- merge(type_year, how_many_left, by = c( "X_JobType2009", "Year"), all = TRUE)

type_year[is.na(type_year)] <- 0
type_year<- type_year %>%
  mutate(yearly_turnover_type =type_turnover/type_emp_pop)

#######
type_year %>% head
# merge back to main data frame... how? 
# create a flatter data set... cast 
library(reshape2)
x<-dcast( type_year, X_JobType2009 ~ Year, value.var = "yearly_turnover_type" ) 
x<-x %>% setNames(paste0('Type_Turnover_', names(.)))
x<-rename(x, X_JobType2009 = Type_Turnover_X_JobType2009)
nums <- sapply(x, is.numeric)
x$Future_Type_Turnover<-rowMeans(x[ , nums])

x <- select ( x, X_JobType2009, Future_Type_Turnover)
head(x)
y<-merge(pretrain, x, all.x = TRUE, by = "X_JobType2009") 
pretrain<-y

pretest<-merge(pretest, x, all.x = TRUE, by = "X_JobType2009") 
x<-NULL




# job function future turnover
pretrain$X_JobFunction2009 %>% unique()

pretrain$X_JobType2009 %>% unique()

pretrain$year_left<- dmy(pretrain$Y_ExitDate) %>% year %>% as.numeric()

year_left_data <- dplyr::filter(pretrain, !is.na(year_left)) 

how_many_left <- dplyr::filter(pretrain, !is.na(year_left))  %>% 
  group_by(X_JobFunction2009, Year =year_left ) %>%
  summarise(Job_fun_turnover =n())

how_many_total <- dplyr::filter(pretrain)  %>% 
  group_by(X_JobFunction2009 ) %>%
  summarise(Job_fun_pop=n())

job_fun_year <- dplyr::filter(pretrain)  %>%
  group_by(X_JobFunction2009 ) %>%
  summarise(n())

pretrain$year_left<-NULL

list1 <- job_fun_year$X_JobFunction2009
list2 <- rep(c(2009, 2010, 2011, 2012, 2013, 2014), each = length(list1))
job_fun_year <- as.data.frame(cbind(list1, list2)) 
names(job_fun_year)<-c("X_JobFunction2009", "Year")

job_fun_year <- merge(job_fun_year, how_many_total, by="X_JobFunction2009", all = TRUE)
job_fun_year<- merge(job_fun_year, how_many_left, by = c( "X_JobFunction2009", "Year"), all = TRUE)

job_fun_year[is.na(job_fun_year)] <- 0
job_fun_year<- job_fun_year %>%
  mutate(yearly_Job_fun_turnover =Job_fun_turnover/Job_fun_pop)

#######
job_fun_year %>% head
# merge back to main data frame... how? 
# create a flatter data set... cast 
library(reshape2)
x<-dcast( job_fun_year, X_JobFunction2009 ~ Year, value.var = "yearly_Job_fun_turnover" ) 
x<-x %>% setNames(paste0('Job_Fun_Turnover_', names(.)))
x<-rename(x, X_JobFunction2009 = Job_Fun_Turnover_X_JobFunction2009)
nums <- sapply(x, is.numeric)
x$Future_Job_Fun_Turnover<-rowMeans(x[ , nums])

x <- select ( x, X_JobFunction2009, Future_Job_Fun_Turnover)
head(x)
y<-merge(pretrain, x, all.x = TRUE, by = "X_JobFunction2009") 
pretrain<-y

pretest<-merge(pretest, x, all.x = TRUE, by = "X_JobFunction2009") 
x<-NULL








#################################
# bring in unemployment info! 

country_unemployment<- read.csv(file = "00_data/country_unemployment.csv", stringsAsFactors = FALSE, na.strings = "")
head(country_unemployment)
names(country_unemployment) <- c("X_Country2009", "Country_Code", "2009","2010","2011","2012","2013", "2014")

str(country_unemployment)

for(i in 1:ncol(country_unemployment)){
  country_unemployment[is.na(country_unemployment[,i]), i] <- mean(country_unemployment[,i], na.rm = TRUE)
}

country_unemployment<-country_unemployment %>% setNames(paste0('Country_Unemploy_', names(.)))

head(country_unemployment)
country_unemployment<-rename(country_unemployment, X_Country2009 = Country_Unemploy_X_Country2009)
country_unemployment<-select(country_unemployment, -Country_Unemploy_Country_Code)
nums <- sapply(country_unemployment, is.numeric)
country_unemployment$Future_country_unemployment<-rowMeans(country_unemployment[ , nums])


pretrain$X_Country2009[!pretrain$X_Country2009 %in% country_unemployment$X_Country2009] %>% unique


pretrain<-merge(pretrain, country_unemployment, all.x = TRUE, by = "X_Country2009") 


#####################################
# pretest
# add country_unemployment

# add un emp
pretest<-merge(pretest, country_unemployment, all.x = TRUE, by = "X_Country2009") 
#pretest<-merge(pretest, x, all.x = TRUE, by = "X_Country2009") 


rm(list=setdiff(ls(), c("pretrain", "pretest"))) 
