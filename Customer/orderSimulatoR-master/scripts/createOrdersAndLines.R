createOrdersAndLines <- function(n = 1500, maxLines = 30, rate = 0.8) {
        
        # First step in generating orders
        # Generates a list of orders and order lines, and returns a data frame
        
        # Requires:
        #################################################
        
        # n: Number of orders to generate
        # maxLines: Maximum number of lines on an order. Note this cannot exceed number of product ids.
        # rate: Uses function 1/i^rate to generate discrete probability for lines on an order, where 
        # i = 1:maxLines and rate manipulates the likelihood of seeing lower value lines versus higher value lines.
        # Values greater than zero cause a distribution weighted to lower line counts on each order. 
        # Values less than zero cause the distribution to be more heavily weighted to larger line counts.
        
        
        # Code:
        #################################################
        
        # Generate discrete probabilities for line counts
        lineProb <- NULL
        for (i in 1:maxLines) {
                lineProb[i] <- 1/i^rate         
        }
        lineProb <- lineProb/sum(lineProb)
        
        # Generate unique order id's
        order.id.unique <- seq(1:n)    
        
        # Generate order line counts
        set.seed(100)   # For reproducibility
        order.lines.count <- sample(x = 1:maxLines, 
                                    size = n, 
                                    replace = T,
                                    prob = lineProb) 
        
        # Generate list of order id's 
        order.id <- NULL
        for (i in 1:n) {
                order.id <- c(order.id, 
                              rep(order.id.unique[i], order.lines.count[i]))
                              
        }
        
        # Generate list of order lines
        order.line <- NULL
        for (i in 1:n) {
                for (j in 1:order.lines.count[i]) {
                        order.line <- c(order.line,
                                        j)
                }
        }
        
        # Combine order.id and order.line in to dataframe
        orders <- as.data.frame(cbind(order.id, order.line))
        orders
        
}

