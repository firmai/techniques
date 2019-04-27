# Simplest metrorail example using amplpy, Gurobi and ticdat

from amplpy import AMPL
from ticdat import TicDatFactory, standard_main, ampl_format
from itertools import product

input_schema = TicDatFactory (
    parameters=[["Key"], ["Value"]],
    load_amounts=[["Amount"],[]],
    number_of_one_way_trips=[["Number"],[]],
    amount_leftover=[["Amount"], []])

default_parameters = {"One Way Price": 2.25, "Amount Leftover Constraint": "Upper Bound"}

solution_schema = TicDatFactory(
    load_amount_details=[["Number One Way Trips", "Amount Leftover", "Load Amount"],
                           ["Number Of Visits"]],
    load_amount_summary=[["Number One Way Trips", "Amount Leftover"],["Number Of Visits"]])

def solve(dat):
    assert input_schema.good_tic_dat_object(dat)

    # use default parameters, unless they are overridden by user-supplied parameters
    full_parameters = dict(default_parameters, **{k:v["Value"] for k,v in dat.parameters.items()})

    sln = solution_schema.TicDat() # create an empty solution

    # copy the load_amounts table to an amplpy.DataFrame object
    ampl_dat = input_schema.copy_to_ampl(dat, excluded_tables=
                   set(input_schema.all_tables).difference({"load_amounts"}))

    # solve a distinct MIP for each pair of (# of one-way-trips, amount leftover)
    for number_trips, amount_leftover in product(dat.number_of_one_way_trips, dat.amount_leftover):

        # create and AMPL object and load it with .mod code
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

        # set the AMPL object with the amplpy.DataFrame data and solve
        input_schema.set_ampl_data(ampl_dat, ampl, {"load_amounts": "LOAD_AMTS"})
        ampl.solve()

        if ampl.getValue("solve_result") != "infeasible":
            # store the results if and only if the model is feasible
            for la,x in ampl.getVariable("Num_Visits").getValues().toDict().items():
                if round(x[0]) > 0:
                    sln.load_amount_details[number_trips, amount_leftover, la] = round(x[0])
                    sln.load_amount_summary[number_trips, amount_leftover]["Number Of Visits"]\
                       += round(x[0])
    return sln

# when run from the command line, will read/write json/xls/csv/db/sql/mdb files
if __name__ == "__main__":
    standard_main(input_schema, solution_schema, solve)