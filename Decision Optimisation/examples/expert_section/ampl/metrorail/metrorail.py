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

# this version of the file uses amplpy and Gurobi
from amplpy import AMPL
from ticdat import TicDatFactory, standard_main, ampl_format
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
        return value  in ["Equality", "Upper Bound"]

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

    ampl_dat = input_schema.copy_to_ampl(dat, excluded_tables=
                   set(input_schema.all_tables).difference({"load_amounts"}))
    # solve a distinct MIP for each pair of (# of one-way-trips, amount leftover)
    for number_trips, amount_leftover in product(dat.number_of_one_way_trips, dat.amount_leftover):

        ampl = AMPL()
        ampl.setOption('solver', 'gurobi')
        # use the ampl_format function for AMPL friendly key-named text substitutions
        ampl.eval(ampl_format("""
        set LOAD_AMTS;
        var Num_Visits {LOAD_AMTS} integer >= 0;
        var Amt_Leftover >= {{amount_leftover_lb}}, <= {{amount_leftover_ub}};
        minimize Total_Visits:
           sum {la in LOAD_AMTS} Num_Visits[la];
        subj to Set_Amt_Leftover:
           Amt_Leftover = sum {la in LOAD_AMTS} la * Num_Visits[la] - {{one_way_price}} * {{number_trips}};""",
            number_trips=number_trips, one_way_price=full_parameters["One Way Price"],
            amount_leftover_lb=amount_leftover if full_parameters["Amount Leftover Constraint"] == "Equality" else 0,
            amount_leftover_ub=amount_leftover))

        input_schema.set_ampl_data(ampl_dat, ampl, {"load_amounts": "LOAD_AMTS"})
        ampl.solve()

        if ampl.getValue("solve_result") != "infeasible":
            # store the results if and only if the model is feasible
            for la,x in ampl.getVariable("Num_Visits").getValues().toDict().items():
                if round(x) > 0:
                    sln.load_amount_details[number_trips, amount_leftover, la] = round(x)
                    sln.load_amount_summary[number_trips, amount_leftover]["Number Of Visits"]\
                       += round(x)
    return sln
# ---------------------------------------------------------------------------------

# ------------------------ provide stand-alone functionality ----------------------
# when run from the command line, will read/write json/xls/csv/db/sql/mdb files
if __name__ == "__main__":
    standard_main(input_schema, solution_schema, solve)
# ---------------------------------------------------------------------------------
