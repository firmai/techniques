createProductQuantities <- function(orders, maxQty = 50, rate = 1.2) {
        
        # Fifth and final step in order creation
        # Creates quantities of products on each line
        
        # Requires:
        #################################################
        
        # orders: created from previous four steps
        # maxQty: Maximum quantity of products on an order. 
        # rate: Uses function 1/i^rate to generate discrete probability for quantity of products on a line, where 
        # i = 1:maxQty and rate manipulates the likelihood of seeing lower quantities versus higher quantities.
        # Values greater than zero cause a distribution weighted to lower quantitis on each line. 
        # Values less than zero cause the distribution to be more heavily weighted to larger quantities on each line.
        
        require(dplyr)
        
        
        # Code:
        #################################################
        
        # Code:
        #################################################
        
        # Generate discrete probabilities for line quantities
        qtyProb <- NULL
        for (i in 1:maxQty) {
                qtyProb[i] <- 1/i^rate         
        }
        qtyProb <- qtyProb/sum(qtyProb)
        
        # Generate order line quantities
        set.seed(100)   # For reproducibility
        quantity <- sample(x = 1:maxQty, 
                                  size = nrow(orders), 
                                  replace = T,
                                  prob = qtyProb)
        
        # Combine orders and order.lines.qty into dataframe
        orders <- cbind(orders, quantity)
        orders
        
}