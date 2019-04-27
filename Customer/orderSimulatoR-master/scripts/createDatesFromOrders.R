createDatesFromOrders <- function(orders,
                                  monthlyOrderDist = c(0.045, 
                                                        0.075, 
                                                        0.100, 
                                                        0.110, 
                                                        0.120, 
                                                        0.125, 
                                                        0.100, 
                                                        0.085, 
                                                        0.075, 
                                                        0.060, 
                                                        0.060, 
                                                        0.045),   # Seasonality of orders 
                                  yearlyOrderDist = c(0.25, 0.325, 0.425),   # Growth of orders in each year
                                  startYear = 2010
                                  ) {
        
        # Second step in order creation
        # Computes dates from the orders template. 
        
        # Requires:
        #################################################
        
        # orders: a dataframe of orders after performing Step 1
        # monthlyOrderDist: a vector of distributions (length = 12, sum = 1) that 
        # indicate the distribution of orders received in each month of the year 
        # yearlyOrderDist: a vector of distributions (length= optional, sum = 1) that
        # indicate the fluctuations in order over successive years. The length is used
        # to determine number of years that orders span
        # startYear: the year of the beginning of orders
        
        require(dplyr)
        require(tidyr)
        require(lubridate)
        
        
        # Code:
        #################################################
        
        # Create distributions for random order dates. 
        endYear <- startYear + (length(yearlyOrderDist)-1)
        years <- seq(from = startYear, to = endYear)
        
        
        # Get list of orders and count of orders
        orderList <- unique(orders$order.id)
        orderCount <- length(orderList)
        
        # Create random dates
        set.seed(100)   # Needed for reproducible samples of month and years
        randomMonths <- sample(x=seq(1,12), size=orderCount, replace=TRUE, prob=monthlyOrderDist)
        randomYears <- sample(x=years, size=orderCount, replace=TRUE, prob=yearlyOrderDist)
        randomDays <- sample(x=seq(1,31), size=orderCount, replace=TRUE)
        
        # Combine random days, months and years
        randomDates <- as.data.frame(cbind(randomDays, randomMonths, randomYears))
        names(randomDates) <- c("Day", "Month", "Year")
        
        # Check if random Day exceeds last day of month; if so, replace with last day of month
        randomDates <- randomDates %>%
                mutate(FirstOfMonth = ymd(paste(Year, Month, "01"))) %>%
                mutate(LastOfMonth = ceiling_date(FirstOfMonth + days(1), "month") - days(1)) %>%
                mutate(MaxDay = day(LastOfMonth)) %>%
                mutate(Day = ifelse(Day > MaxDay, MaxDay, Day))
        
        # Get random date and day of week
        randomDates <- randomDates %>%
                mutate(Date = ymd(paste(Year, Month, Day))) %>%
                mutate(DOW = wday(Date))
        
        # Take care of weekends which are non-working days
        for (i in 1:orderCount){
                # Sundays
                if (randomDates[[i, 8]] == 1 && randomDates[[i,1]] <= 15){
                        randomDates[[i, 1]] <-  randomDates[[i, 1]] + (2 + i%%2) # Split between Monday and Tues
                }
                if (randomDates[[i, 8]] == 1 && randomDates[[i,1]] > 15){
                        randomDates[[i, 1]] <-  randomDates[[i, 1]] - (4 + i%%2) # Split between Thurs and Fri
                }
                # Saturdays
                if (randomDates[[i, 8]] == 7 && randomDates[[i, 1]] >= 15){
                        randomDates[[i, 1]] <-  randomDates[[i, 1]] - (3 + i%%2) # Split between Tues and Wed
                }
                if (randomDates[[i, 8]] == 7 && randomDates[[i, 1]] < 15){
                        randomDates[[i, 1]] <-  randomDates[[i, 1]] + (5 + i%%2) # Split between Thurs and Fri  
                }
        }
        randomDates <- randomDates %>%
                mutate(Date2 = ymd(paste(Year, Month, Day))) %>%
                mutate(DOW2 = wday(Date2))
        
        # Select date column only, and order dates in chronological order
        dates <- randomDates %>%
                select(Date2) %>%
                arrange(Date2) %>%
                rename(Date = Date2)
        
        # Join dates with unique orders
        order.dates <- as.data.frame(cbind(orderList, dates))
        order.dates <- rename(order.dates, order.id = orderList, order.date = Date)
        
        # Add dates to template
        ordersAndLines <- orders %>%
                select(order.id, order.line) %>%
                left_join(order.dates, by = "order.id")
        
        # Return
        ordersAndLines
        
}