assignCustomersToOrders <- function(orders, customers, rate = 0.6) {
        
        # Third step in order creation
        # Assigns customers to order using a customer-line frequency distribution
        
        # Requires:
        #################################################
        
        # orders: a dataframe of orders after performing the previous two steps
        # customers: a data frame of customers with ids in the first column
        # rate: Rate of probability of customer-orders, uses 1/i^rate to create 
        # discreate probability where i is range of customers
        
        
        require(dplyr)
        
        
        # Code:
        #################################################
        
        # Get customer ids and number of customers
        customer.id <- customers[,1]    # Customer id vector
        n = length(customer.id)     # Number of customers
        
        # Shuffle customer ids
        set.seed(100)
        customer.id.random <- sample(customer.id)
        
        # Generate distribution for customer order-line frequency
        custProb <- NULL
        for (i in 1:n) {
                custProb[i] <- 1/i^rate
        }
        custProb <- custProb/sum(custProb)
        
        # Sample random customers using the customer distribution
        order.id.unique <- unique(orders$order.id)
        set.seed(101)
        customerOrderAssignment <- sample(x = customer.id.random, 
                                          size = length(order.id.unique), 
                                          replace = T,
                                          prob = custProb)
        
        # Combine order-customer assignment for left_join
        orderCustomerDF <- as.data.frame(cbind(order.id.unique, customerOrderAssignment))
        orderCustomerDF <- rename(orderCustomerDF, 
                                  order.id = order.id.unique, 
                                  customer.id = customerOrderAssignment)
        
        # Merge order-customer assigment with orders by order.id
        orders <- left_join(orders, orderCustomerDF)
        
}