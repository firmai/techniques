#!/usr/bin/python

# Copyright 2017 Opalytics, Inc.
#

# Solves a simple product-promotion optimization problem. The objective is to maximize
# the total expected sales of a series of products. In order to prevent cannibalization,
# there are restrictions on the total number of promotions for a given product family.
# There is also a restriction on the total investment (i.e. probable loss of revenue)
# across the entire solution as a whole.

# Implement core functionality needed to achieve modularity.
# 1. Define the input data schema
# 2. Define the output data schema
# 3. Create a solve function that accepts a data set consistent with the input
#    schema and (if possible) returns a data set consistent with the output schema.
#
# Provides command line interface via ticdat.standard_main
# For example, typing
#   python soda_promotion_optimizer.py -i input_data.xlsx -o solution_data.xlsx
# will read from a model stored in the file input_data.xlsx and write the solution
# to solution_data.xlsx.

from ticdat import TicDatFactory, standard_main, Slicer, gurobi_env
import gurobipy as gu

input_schema = TicDatFactory(parameters = [["Key"],["Value"]],
                             products = [["Name"],["Family"]],
                             forecast_sales = [["Product", "Cost Per Unit"],
                                               ["Sales"]],
                             max_promotions = [["Product Family"],["Max Promotions"]])

input_schema.set_data_type("parameters", "Key", number_allowed=False, strings_allowed=("Maximum Total Investment",))
input_schema.set_data_type("parameters", "Value", min=0, max=float("inf"), inclusive_max=True)
input_schema.set_data_type("forecast_sales", "Cost Per Unit", min=0, max=float("inf"))
input_schema.set_data_type("max_promotions", "Product Family", number_allowed=False, strings_allowed= ("Dark", "Clear"))
input_schema.set_data_type("max_promotions", "Max Promotions", min=0, max=float("inf"), inclusive_max=False,
                           must_be_int=True)

input_schema.add_foreign_key("products", "max_promotions", ["Family", "Product Family"])
input_schema.add_foreign_key("forecast_sales", "products", ["Product", "Name"])


solution_schema  = TicDatFactory(parameters = [["Key"],["Value"]],
                                 product_pricing = [["Product"],["Cost Per Unit", "Family", "Promotional Status"]])

def solve(dat):
    assert input_schema.good_tic_dat_object(dat)
    assert not input_schema.find_foreign_key_failures(dat)
    assert not input_schema.find_data_type_failures(dat)

    normal_price = {pdct:0 for pdct in dat.products}
    for pdct, price in dat.forecast_sales:
        normal_price[pdct] = max(normal_price[pdct], price)

    def revenue(pdct, price):
        return dat.forecast_sales[pdct, price]["Sales"] * price
    def investment(pdct, price):
        return max(0, dat.forecast_sales[pdct, price]["Sales"] * normal_price[pdct] -
                      revenue(pdct, normal_price[pdct]))
    mdl = gu.Model("soda promotion", env=gurobi_env())

    pdct_price = mdl.addVars(dat.forecast_sales, vtype=gu.GRB.BINARY,name='pdct_price')

    mdl.addConstrs((pdct_price.sum(pdct,'*') == 1 for pdct in dat.products), name = "pick_one_price")

    total_qty = mdl.addVar(name="total_qty")
    total_revenue = mdl.addVar(name="total_revenue")
    total_investment = mdl.addVar(name="total_investment", ub=dat.parameters["Maximum Total Investment"]["Value"]
                                if "Maximum Total Investment" in dat.parameters else float("inf"))

    mdl.addConstr(total_qty == pdct_price.prod({_:dat.forecast_sales[_]["Sales"] for _ in pdct_price}))
    mdl.addConstr(total_revenue == pdct_price.prod({_:revenue(*_) for _ in pdct_price}))
    mdl.addConstr(total_investment == pdct_price.prod({_:investment(*_) for _ in pdct_price}))

    pf_slice = Slicer((dat.products[pdct]["Family"], pdct, price) for pdct, price in pdct_price)
    for pdct_family, r in dat.max_promotions.items():
        mdl.addConstr(gu.quicksum(pdct_price[_pdct, _price]
                                  for _pdct_family, _pdct, _price in pf_slice.slice(pdct_family, '*', '*')
                                  if _price != normal_price[_pdct]) <= r["Max Promotions"],
                      name = "max_promotions_%s"%pdct_family)

    mdl.setObjective(total_qty, sense=gu.GRB.MAXIMIZE)
    mdl.optimize()

    if mdl.status == gu.GRB.OPTIMAL:
        sln = solution_schema.TicDat()
        for (pdct, price), var in pdct_price.items():
            if abs(var.X -1) < 0.0001: # i.e. almost one
                sln.product_pricing[pdct] = [price, dat.products[pdct]["Family"],
                                             "Normal Price" if price == normal_price[pdct] else "Discounted"]
        sln.parameters["Total Quantity Sold"] = total_qty.X
        sln.parameters["Total Revenue"] = total_revenue.X
        sln.parameters["Total Investment"] = total_investment.X
        number_meaningful_discounts = 0
        for (pdct, price), r in dat.forecast_sales.items():
            if (price < normal_price[pdct] and
                r["Sales"] > dat.forecast_sales[pdct,normal_price[pdct]]["Sales"]):
                number_meaningful_discounts += 1
        sln.parameters["Number of Meaningful Discounts"] = number_meaningful_discounts

        return sln

if __name__ == "__main__":
    standard_main(input_schema, solution_schema, solve)
