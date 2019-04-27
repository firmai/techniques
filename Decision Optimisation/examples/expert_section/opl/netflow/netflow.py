#!/usr/bin/python

# Copyright 2015, 2016 Opalytics, Inc.

# Solve a multi-commodity flow problem as python package.

# Implement core functionality needed to achieve modularity.
# 1. Define the input data schema
# 2. Define the output data schema
# 3. Create a solve function that accepts a data set consistent with the input
#    schema and (if possible) returns a data set consistent with the output schema.
#
# Provides command line interface via ticdat.standard_main
# For example, typing
#   python netflow.py -i csv_data -o solution_csv_data
# will read from a model stored in .csv files in the csv_data directory
# and write the solution to .csv files in the solution_csv_data directory
#
# Note that file requires netflow.mod to be in the same directory

from ticdat import TicDatFactory, standard_main, opl_run

# ------------------------ define the input schema --------------------------------
input_schema = TicDatFactory(
    commodities=[["Name"], []],
    nodes=[["Name"], []],
    arcs=[["Source", "Destination"], ["Capacity"]],
    cost=[["Commodity", "Source", "Destination"], ["Cost"]],
    inflow=[["Commodity", "Node"], ["Quantity"]]
)

# add foreign key constraints
input_schema.add_foreign_key("arcs", "nodes", ['Source', 'Name'])
input_schema.add_foreign_key("arcs", "nodes", ['Destination', 'Name'])
input_schema.add_foreign_key("cost", "nodes", ['Source', 'Name'])
input_schema.add_foreign_key("cost", "nodes", ['Destination', 'Name'])
input_schema.add_foreign_key("cost", "commodities", ['Commodity', 'Name'])
input_schema.add_foreign_key("inflow", "commodities", ['Commodity', 'Name'])
input_schema.add_foreign_key("inflow", "nodes", ['Node', 'Name'])

input_schema.set_data_type("arcs", "Capacity",  max=float("inf"),
                           inclusive_max=True)
input_schema.set_data_type("cost", "Cost")
input_schema.set_data_type("inflow", "Quantity", min=-float("inf"),
                          inclusive_min=False)
# ---------------------------------------------------------------------------------

# ------------------------ define the output schema -------------------------------
solution_schema = TicDatFactory(
    flow=[["Commodity", "Source", "Destination"], ["Quantity"]],
    parameters=[["Key"], ["Value"]])
# ---------------------------------------------------------------------------------

# ------------------------ solving section-----------------------------------------
def solve(dat):
    """
    core solving routine
    :param dat: a good ticdat for the input_schema
    :return: a good ticdat for the solution_schema, or None
    """
    return opl_run("netflow.mod", input_schema, dat, solution_schema)

# ---------------------------------------------------------------------------------

# ------------------------ provide stand-alone functionality ----------------------
# when run from the command line, will read/write json/xls/csv/db/sql/mdb files
if __name__ == "__main__":
    standard_main(input_schema, solution_schema, solve)
# ---------------------------------------------------------------------------------
