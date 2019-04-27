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
#   python netflow.py -i netflow_sample_data.sql -o solution_data.sql
# will read from a model stored in netflow_sample_data.sql and write a solution to
# solution_data.sql file.
#
# Note that file requires netflow.lng to be in the same directory

from ticdat import TicDatFactory, standard_main, lingo_run

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

    # There are the variables populated by the netflow.lng file.
    solution_variables = TicDatFactory(flow=[["Commodity", "Source", "Destination"], ["Quantity"]])

    sln = lingo_run("netflow.lng", input_schema, dat, solution_variables)
    if sln:
        rtn = solution_schema.TicDat(flow = {k:r for k,r in sln.flow.items() if r["Quantity"] > 0})
        rtn.parameters["Total Cost"] = sum(dat.cost[h, i, j]["Cost"] * r["Quantity"]
                                           for (h, i, j), r in rtn.flow.items())
        return rtn

# ---------------------------------------------------------------------------------

# ------------------------ provide stand-alone functionality ----------------------
# when run from the command line, will read/write xls/csv/db/sql/mdb files
if __name__ == "__main__":
    standard_main(input_schema, solution_schema, solve)
# ---------------------------------------------------------------------------------
