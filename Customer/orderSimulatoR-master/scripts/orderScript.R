# ORDER SIMULATOR SCRIPT
###################################################

# This script can be used as a template to generate orders.  
# The inputs to the order simulation script are the following data sets:
# 1. customers.xlsx: An excel file of 30 customers including the customer.id,
#    customer.name, customer.city, and customer.state
# 2. bikes.xlsx: An excel file of 97 bike models and various product data
# 3. customer_product_interactions.xlsx: An excel file that with a matrix of 
#    probabilities for the likelihood of a customer.id selecting the bike.id.
#    See the excel file for more information on how to create an interaction
#    matrix.

source("./scripts/createOrdersAndLines.R")
source("./scripts/createDatesFromOrders.R")
source("./scripts/assignCustomersToOrders.R")
source("./scripts/assignProductsToCustomerOrders.R")
source("./scripts/createProductQuantities.R")

require(xlsx)

# Read customer, product, and customer-product interaction data
####################################################

customers <- read.xlsx("./data/bikeshops.xlsx", sheetIndex = 1)
products <- read.xlsx("./data/bikes.xlsx", sheetIndex = 1) 
customerProductProbs <- read.xlsx("./data/customer_product_interactions.xlsx", 
                                  sheetIndex = 1, 
                                  startRow = 15)
customerProductProbs <- customerProductProbs[,-(2:11)]  # Remove unnecessary columns


# Create orders
####################################################

# Step 1 - Create orders and lines
orders <- createOrdersAndLines(n = 2000, maxLines = 30, rate = 1)       

# Step 2 - Add dates to the orders
orders <- createDatesFromOrders(orders, 
                                startYear = 2011, 
                                yearlyOrderDist = c(.16, .18, .22, .20, .24),
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
                                                     0.045))

# Step 3 - Assign customer id's to order lines
orders <- assignCustomersToOrders(orders, customers, rate = 0.8)

# Step 4 - Assign product id's to orders based on the customer product probabilities 
orders <- assignProductsToCustomerOrders(orders, customerProductProbs)

# Step 5 - Create product quantities for order lines
orders <- createProductQuantities(orders, maxQty = 10, rate = 3)


# Export order
####################################################

# Warning: this step can take a significant amount of time depending on the dataset size
write.xlsx(orders, file = "./data/orders.xlsx")
