# orderSimulatoR

___Fast and easy `R` order simulation for customer and product learning!___

## About

`orderSimulatoR` enables fast and easy creation of order data for simulation, data mining and machine learning. In it's current form, the `orderSimulatoR` is a collection of scripts that can be used to generate sample order data from the following inputs: customer table (e.g. `bikeshops.xlsx`), products table (e.g. `bikes.xlsx`), and customer-products interaction table (e.g. `customer_product_interactions.xlsx`). The output will be order data. Example input files are provided (refer to the data folder). The output generated is similar to that in the file `orders.xlsx`.

## Why this Helps

It's very difficult to create custom order data for data mining, visualization, trending, etc. I've searched for good data sets, and I came to the conclusion that I'm better off creating my own orders data for messing around with on my blog. In the process, I made an algorithm to generate the orders. I made the algorithm publicly available so others can use to show off their analytical abilities. T

## Creating Orders

The process to create orders (shown below) is fast and easy, and the result is orders with customized trends depending on the inputs you create and the parameters you select. I've provided some sample data in the `data` folder to help with the explanation. The scripts used are in the `scripts` folder. [Click here](http://www.mattdancho.com/business/2016/07/12/orderSimulatoR.html) for an in-depth walkthrough.

<!-- ![Order Simulation Process](/figures/OrderSimProcess.jpg =500x500)  -->

<img src="/figures/OrderSimProcess.jpg" alt="Order Simulation Process" width="650" height="400" align="middle"/>

## Example Usage

* [ORDERSIMULATOR: SIMULATE ORDERS FOR BUSINESS ANALYTICS](http://www.mattdancho.com/business/2016/07/12/orderSimulatoR.html) - This is a walkthrough on how to simulate orders using `orderSimulatoR`.