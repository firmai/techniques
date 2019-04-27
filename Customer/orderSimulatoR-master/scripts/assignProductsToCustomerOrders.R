assignProductsToCustomerOrders <- function(orders, customerProductProbs) {
        
        # Fourth step in order creation
        # Assigns product id's to customer-order-lines
        
        # Requires:
        #################################################
        
        # orders: created from previous three steps
        # custProductProbs: a matrix linking each product.id to customers.ids,
        # with the value in each cell indicating the probability a particular 
        # customer.id selecting a particular product.id
        
        
        require(dplyr)
        
        
        # Process product distributions for each customer
        ####################################################### 
        
        product.id <- customerProductProbs[,1]
        customerProbMatrix <- customerProductProbs[,2:ncol(customerProductProbs)]
        
        # Make sure values in matrix columns sum to one
        customerProbMatrix <- t(t(customerProbMatrix)/colSums(customerProbMatrix))
        
        
        # Assign products to customer-order lines
        #######################################################
        
        # Get number of lines on each customer-order to sample the product list
        customerSamplesNeeded <- orders %>%
                group_by(order.id, customer.id) %>%
                summarise(line.count = n())
        
        # Sample products according to customer-product probabilities
        for (i in 1:nrow(customerSamplesNeeded)) {
                set.seed(i)
                cust.id <- customerSamplesNeeded[[i,2]] # Retreive customer id for probability
                custProbability <- customerProbMatrix[,cust.id]
                customerSamplesNeeded$product.sampling[[i]] <- sample(x = product.id,
                                                                      size = customerSamplesNeeded$line.count[[i]],
                                                                      replace = FALSE,
                                                                      prob = custProbability) 
        }
        
        # Unlist the samples to create a vector of length of order-lines
        product.samples <- as.list(customerSamplesNeeded$product.sampling)
        product.id <- unlist(product.samples)
        
        # Combine the orders with the products selected and return orders
        orders <- cbind(orders, product.id)
        orders
        
}