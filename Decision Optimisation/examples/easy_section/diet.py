# Simplest diet example using gurobipy and ticdat

import gurobipy as gu
from ticdat import TicDatFactory,  standard_main, gurobi_env

input_schema = TicDatFactory (
    categories = [["Name"],["Min Nutrition", "Max Nutrition"]],
    foods  = [["Name"],["Cost"]],
    nutrition_quantities = [["Food", "Category"], ["Quantity"]])

solution_schema = TicDatFactory(
    parameters = [["Key"],["Value"]],
    buy_food = [["Food"],["Quantity"]],
    consume_nutrition = [["Category"],["Quantity"]])

def solve(dat):
    assert input_schema.good_tic_dat_object(dat)

    mdl = gu.Model("diet", env=gurobi_env())

    nutrition = {c:mdl.addVar(lb=n["Min Nutrition"], ub=n["Max Nutrition"], name=c)
                for c,n in dat.categories.items()}

    # Create decision variables for the foods to buy
    buy = {f:mdl.addVar(name=f) for f in dat.foods}

     # Nutrition constraints
    for c in dat.categories:
        mdl.addConstr(gu.quicksum(dat.nutrition_quantities[f,c]["Quantity"] * buy[f]
                      for f in dat.foods) == nutrition[c],
                      name = c)

    mdl.setObjective(gu.quicksum(buy[f] * c["Cost"] for f,c in dat.foods.items()),
                     sense=gu.GRB.MINIMIZE)
    mdl.optimize()

    if mdl.status == gu.GRB.OPTIMAL:
        sln = solution_schema.TicDat()
        for f,x in buy.items():
            if x.x > 0:
                sln.buy_food[f] = x.x
        for c,x in nutrition.items():
            sln.consume_nutrition[c] = x.x
        sln.parameters['Total Cost'] = sum(dat.foods[f]["Cost"] * r["Quantity"]
                                           for f,r in sln.buy_food.items())
        return sln

# when run from the command line, will read/write json/xls/csv/db/sql/mdb files
if __name__ == "__main__":
    standard_main(input_schema, solution_schema, solve)