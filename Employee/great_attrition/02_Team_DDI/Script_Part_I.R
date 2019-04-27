###############################
######SIOP ML Competition######
###########TEAMDDI#############
#######Solution Part I#########
###########Feb 2018############
###############################

# In Part I of the syntax,
# external country-level data is merged with the original data
# and meta variables are created

### IMPORT LIBRARIES ###
library(plyr)
library(dplyr)
library(readxl)


## Read in data
df_train <- read.csv("TrainingSet.csv", stringsAsFactors = F)
df_test <- read.csv("TestSet.csv", stringsAsFactors = F)

df <- rbind.fill(df_train, df_test)


### META VARIABLES ###
# Number of supervisors
# Whether the individual is supervisor or not 

## Count # of supervisors

names(df)
supervisors <- c("S_SupervisorGlobal_ID2009",
                 "S_SupervisorGlobal_ID2008",
                 "S_SupervisorGlobal_ID2007",
                 "S_SupervisorGlobal_ID2006",      
                 "S_SupervisorGlobal_ID2005",
                 "S_SupervisorGlobal_ID2004") 

num_supers <- df[,supervisors] %>%
  rowwise()%>%
  do(data.frame(., Count = n_distinct(unlist(.))))

df$NumberofSupervisors <- num_supers$Count

## Are you a supervisor?

# All of the supervisors in the data, contains dupes
all_supers <- unlist(df[supervisors])

# Lists each supervisor once
unique_supers <- unique(all_supers)


# 0 = not a supervisor in this data, 1 = a supervisor at some point
# Note, we're assuming this isn't an exhaustive list of all employees.
# Some of the participants who are marked as 0 may have been a supervisor
# at some point. 

df$EveraSupervisor <- as.numeric(df$Global_ID %in% unique_supers)

table(df$EveraSupervisor)



### MERGE COUNTRY-LEVEL DATA ###
# Country-level unemployment rate (https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS)
# Consumer Confidence Index (https://data.oecd.org/leadind/consumer-confidence-index-cci.htm)
# Composite Leading Indicator (https://data.oecd.org/leadind/composite-leading-indicator-cli.htm#indicator-chart)

country <- lapply(excel_sheets("MLComp_CountryData.xlsx"), read_excel, path = "MLComp_CountryData.xlsx")

# names(country) <- excel_sheets("MLComp_CountryData.xlsx")

for (i in 1:length(country)){
  temp_df <- as.data.frame(country[i])

  assign(paste0(excel_sheets("MLComp_CountryData.xlsx")[i]), temp_df)
}

df <- left_join(df, TotalUnemp2009, by = "X_Country2009")
df <- left_join(df, TotalUnemp2008 , by = "X_Country2008")
df <- left_join(df, TotalUnemp2007 , by = "X_Country2007")
df <- left_join(df, TotalUnemp2006 , by = "X_Country2006")
df <- left_join(df, TotalUnemp2005 , by = "X_Country2005")
df <- left_join(df, TotalUnemp2004 , by = "X_Country2004")
df <- left_join(df, CCI2009 , by = "X_Country2009")
df <- left_join(df, CCI2008 , by = "X_Country2008")
df <- left_join(df, CCI2007 , by = "X_Country2007")
df <- left_join(df, CCI2006 , by = "X_Country2006")
df <- left_join(df, CCI2005 , by = "X_Country2005")
df <- left_join(df, CCI2004 , by = "X_Country2004")
df <- left_join(df, CLI2009 , by = "X_Country2009")
df <- left_join(df, CLI2008 , by = "X_Country2008")
df <- left_join(df, CLI2007 , by = "X_Country2007")
df <- left_join(df, CLI2006 , by = "X_Country2006")
df <- left_join(df, CLI2005 , by = "X_Country2005")
df <- left_join(df, CLI2004 , by = "X_Country2004")

# Save merged dataset
write.csv(df, "Merged.csv")
