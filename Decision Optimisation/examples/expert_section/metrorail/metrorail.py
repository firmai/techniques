#
# Models Tallys Yunes Metrorail tickets problem.
# https://orbythebeach.wordpress.com/2018/03/01/buying-metrorail-tickets-in-miami/
#
# Implement core functionality needed to achieve modularity.
# 1. Define the input data schema
# 2. Define the output data schema
# 3. Create a solve function that accepts a data set consistent with the input
#    schema and (if possible) returns a data set consistent with the output schema.
#
# Provides command line interface via ticdat.standard_main
# For example, typing
#   python metrorail.py -i metrorail_sample_data.json -o metrorail_solution_data.json
# will read from a model stored in the file metrorail_sample_data.json and write the
# solution to metrorail_solution_data.json.

# this version of the file uses Gurobi
import gurobipy as gu
from ticdat import TicDatFactory, standard_main
from itertools import product

# ------------------------ define the input schema --------------------------------
input_schema = TicDatFactory (
    parameters=[["Key"], ["Value"]],
    load_amounts=[["Amount"],[]],
    number_of_one_way_trips=[["Number"],[]],
    amount_leftover=[["Amount"], []])

input_schema.set_data_type("load_amounts", "Amount", min=0, max=float("inf"),
                           inclusive_min=False, inclusive_max=False)

input_schema.set_data_type("number_of_one_way_trips", "Number", min=0, max=float("inf"),
                           inclusive_min=False, inclusive_max=False, must_be_int=True)

input_schema.set_data_type("amount_leftover", "Amount", min=0, max=float("inf"),
                           inclusive_min=True, inclusive_max=False)


default_parameters = {"One Way Price": 2.25, "Amount Leftover Constraint": "Upper Bound"}
def _good_parameter_key_value(key, value):
    if key == "One Way Price":
        try:
            return 0 < value < float("inf")
        except:
            return False
    if key == "Amount Leftover Constraint":
        return value  in ["Equality", "Upper Bound", "Upper Bound With Leftover Multiple Rule"]

assert all(_good_parameter_key_value(k,v) for k,v in default_parameters.items())

input_schema.set_data_type("parameters", "Key", number_allowed=False,
                           strings_allowed=default_parameters)
input_schema.add_data_row_predicate("parameters", predicate_name="Good Parameter Value for Key",
    predicate=lambda row : _good_parameter_key_value(row["Key"], row["Value"]))
# ---------------------------------------------------------------------------------


# ------------------------ define the output schema -------------------------------
solution_schema = TicDatFactory(
    load_amount_details=[["Number One Way Trips", "Amount Leftover", "Load Amount"],
                           ["Number Of Visits"]],
    load_amount_summary=[["Number One Way Trips", "Amount Leftover"],["Number Of Visits"]])
# ---------------------------------------------------------------------------------


# ------------------------ create a solve function --------------------------------
def solve(dat):
    """
    core solving routine
    :param dat: a good ticdat for the input_schema
    :return: a good ticdat for the solution_schema, or None
    """
    assert input_schema.good_tic_dat_object(dat)
    assert not input_schema.find_foreign_key_failures(dat)
    assert not input_schema.find_data_type_failures(dat)
    assert not input_schema.find_data_row_failures(dat)
    # use default parameters, unless they are overridden by user-supplied parameters
    full_parameters = dict(default_parameters, **{k:v["Value"] for k,v in dat.parameters.items()})

    sln = solution_schema.TicDat() # create an empty solution'

    # solve a distinct MIP for each pair of (# of one-way-trips, amount leftover)
    for number_trips, amount_leftover in product(dat.number_of_one_way_trips, dat.amount_leftover):

        mdl = gu.Model("metrorail")

        # Create decision variables
        number_vists = {la:mdl.addVar(vtype = gu.GRB.INTEGER, name="load_amount_%s"%la)
                        for la in dat.load_amounts}
        amount_leftover_var = mdl.addVar(name="amount_leftover", lb=0, ub=amount_leftover)

        # an equality constraint is modeled here as amount_leftover_var.lb = amount_leftover_var.ub
        if full_parameters["Amount Leftover Constraint"] == "Equality":
            amount_leftover_var.lb = amount_leftover
        # for left-over is multiple, we will still respect the amount leftover upper bound
        # but will also enforce that the amount leftover is a multiple of the one way price
        if full_parameters["Amount Leftover Constraint"] == "Upper Bound With Leftover Multiple Rule":
            leftover_multiple = mdl.addVar(vtype = gu.GRB.INTEGER, name="leftover_multiple")
            mdl.addConstr(amount_leftover_var == full_parameters["One Way Price"] * leftover_multiple,
                          name="set_leftover_multiple")

        # add a constraint to set the amount leftover
        mdl.addConstr(amount_leftover_var ==
                      gu.quicksum(la * number_vists[la] for la in dat.load_amounts) -
                      full_parameters["One Way Price"] * number_trips,
                      name="set_amount_leftover")


        # minimize the total number of visits to the ticket office
        mdl.setObjective(gu.quicksum(number_vists.values()), sense=gu.GRB.MINIMIZE)
        mdl.optimize()

        if mdl.status in [gu.GRB.OPTIMAL, gu.GRB.SUBOPTIMAL]:
            # store the results if and only if the model is feasible
            for la,x in number_vists.items():
                if round(x.x) > 0:
                    sln.load_amount_details[number_trips, amount_leftover, la] = round(x.x)
                    sln.load_amount_summary[number_trips, amount_leftover]["Number Of Visits"]\
                       += round(x.x)
    return sln
# ---------------------------------------------------------------------------------

# ------------------------ provide stand-alone functionality ----------------------
# when run from the command line, will read/write json/xls/csv/db/sql/mdb files
if __name__ == "__main__":
    standard_main(input_schema, solution_schema, solve)
# ---------------------------------------------------------------------------------
