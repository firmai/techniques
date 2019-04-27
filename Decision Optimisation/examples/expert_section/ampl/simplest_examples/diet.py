# Simplest diet example using amplpy, Gurobi and ticdat

from amplpy import AMPL
from ticdat import TicDatFactory, standard_main

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

    # copy the data over to amplpy.DataFrame objects, renaming the data fields as needed
    dat = input_schema.copy_to_ampl(dat, field_renamings={("foods", "Cost"): "cost",
            ("categories", "Min Nutrition"): "n_min", ("categories", "Max Nutrition"): "n_max",
            ("nutrition_quantities", "Quantity"): "amt"})

    # create and AMPL object and load it with .mod code
    ampl = AMPL()
    ampl.setOption('solver', 'gurobi')
    ampl.eval("""
    set CAT;
    set FOOD;

    param cost {FOOD} > 0;

    param n_min {CAT} >= 0;
    param n_max {i in CAT} >= n_min[i];

    param amt {FOOD, CAT} >= 0;

    var Buy {j in FOOD} >= 0;
    var Consume {i in CAT } >= n_min [i], <= n_max [i];

    minimize Total_Cost:  sum {j in FOOD} cost[j] * Buy[j];

    subject to Diet {i in CAT}:
       Consume[i] =  sum {j in FOOD} amt[j,i] * Buy[j];
    """)

    # set the AMPL object with the amplpy.DataFrame data and solve
    input_schema.set_ampl_data(dat, ampl, {"categories": "CAT", "foods": "FOOD"})
    ampl.solve()

    if ampl.getValue("solve_result") != "infeasible":
        sln = solution_schema.copy_from_ampl_variables(
            {("buy_food", "Quantity"):ampl.getVariable("Buy"),
            ("consume_nutrition", "Quantity"):ampl.getVariable("Consume")})
        sln.parameters['Total Cost'] = ampl.getObjective('Total_Cost').value()

        return sln

# when run from the command line, will read/write xls/csv/db/sql/mdb/json files
if __name__ == "__main__":
    standard_main(input_schema, solution_schema, solve)