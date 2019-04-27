#!/usr/bin/python

# Copyright 2017 Opalytics, Inc.
#

# Solves the oil blending problem defined at https://goo.gl/kqXmQE

# Implement core functionality needed to achieve modularity.
# 1. Define the input data schema
# 2. Define the output data schema
# 3. Create a solve function that accepts a data set consistent with the input
#    schema and (if possible) returns a data set consistent with the output schema.
#
# Provides command line interface via ticdat.standard_main
# For example, typing
#   python oil_blend.py -i input_data.xlsx -o solution_data.xlsx
# will read from a model stored in the file input_data.xlsx and write the solution
# to solution_data.xlsx.

# Note that this file requires oil_blend.mod to be in the same directory

from ticdat import TicDatFactory, standard_main, opl_run

# ------------------------ define the input schema --------------------------------
input_schema = TicDatFactory (
     parameters = [["Key"],["Value"]],
     gas = [["Name"],["Demand","Sales Price","Min Octane Rating", "Max Lead Contents"]],
     oil = [["Name"],["Supply","Purchase Price","Octane Rating", "Lead Contents"]])

# no foreign keys

input_schema.set_data_type("parameters", "Key", number_allowed = False,
                           strings_allowed= ["Maximum Production", "Production Cost"])
input_schema.set_data_type("parameters", "Value", min=0, max=float("inf"),
                           inclusive_min=True, inclusive_max=True)
input_schema.set_data_type("gas", "Demand", min=0, max=float("inf"),
                           inclusive_min=True, inclusive_max=False)
input_schema.set_data_type("gas", "Sales Price", min=0, max=float("inf"),
                           inclusive_min=True, inclusive_max=False)
input_schema.set_data_type("gas", "Min Octane Rating", min=0, max=float("inf"),
                           inclusive_min=True, inclusive_max=False)
input_schema.set_data_type("gas", "Max Lead Contents", min=0, max=float("inf"),
                           inclusive_min=True, inclusive_max=True)
input_schema.set_data_type("oil", "Supply", min=0, max=float("inf"),
                           inclusive_min=True, inclusive_max=True)
input_schema.set_data_type("oil", "Purchase Price", min=0, max=float("inf"),
                           inclusive_min=True, inclusive_max=False)
input_schema.set_data_type("oil", "Octane Rating", min=0, max=float("inf"),
                           inclusive_min=True, inclusive_max=False)
input_schema.set_data_type("oil", "Lead Contents", min=0, max=float("inf"),
                           inclusive_min=True, inclusive_max=False)
# We tolerate float("inf") as the Maximum Production but not the Production Cost
input_schema.add_data_row_predicate("parameters",
                                    lambda row : not (row["Key"] == "Production Cost" and
                                                      row["Value"] == float("inf")))

input_schema.opl_prepend = "inp_" # avoid table name collisions
# ---------------------------------------------------------------------------------


# ------------------------ define the output schema -------------------------------
solution_schema = TicDatFactory(
    parameters = [["Key"],["Value"]],
    advertising = [["Gas"],["Dollars Spent"]],
    blending = [["Oil","Gas"],["Quantity"]])

solution_schema.opl_prepend = "sln_" # avoid table name collisions
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

    return opl_run("oil_blend.mod", input_schema, dat, solution_schema)
# ---------------------------------------------------------------------------------

# ------------------------ provide stand-alone functionality ----------------------
# when run from the command line, will read/write json/xls/csv/db/sql/mdb files
if __name__ == "__main__":
    standard_main(input_schema, solution_schema, solve)
# ---------------------------------------------------------------------------------