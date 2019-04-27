#!/usr/bin/python

# Copyright 2015, 2016 Opalytics, Inc.
#
# edited with permission from Gurobi Optimization, Inc.

# Solve a multi-commodity flow problem as python package.
# This version of the file uses pandas for data tables
# but also iterates over table indicies explicitly and uses
# Sloc to perform pandas slicing.

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

import gurobipy as gu
from ticdat import TicDatFactory, Sloc, standard_main
import pandas as pd

# ------------------------ define the input schema --------------------------------
input_schema = TicDatFactory (
     commodities = [["Name"],[]],
     nodes  = [["Name"],[]],
     arcs = [["Source", "Destination"],["Capacity"]],
     cost = [["Commodity", "Source", "Destination"], ["Cost"]],
     inflow = [["Commodity", "Node"],["Quantity"]]
)

# Define the foreign key relationships
input_schema.add_foreign_key("arcs", "nodes", ['Source', 'Name'])
input_schema.add_foreign_key("arcs", "nodes", ['Destination', 'Name'])
input_schema.add_foreign_key("cost", "nodes", ['Source', 'Name'])
input_schema.add_foreign_key("cost", "nodes", ['Destination', 'Name'])
input_schema.add_foreign_key("cost", "commodities", ['Commodity', 'Name'])
input_schema.add_foreign_key("inflow", "commodities", ['Commodity', 'Name'])
input_schema.add_foreign_key("inflow", "nodes", ['Node', 'Name'])

# Define the data types
input_schema.set_data_type("arcs", "Capacity", min=0, max=float("inf"),
                           inclusive_min=True, inclusive_max=True)
input_schema.set_data_type("cost", "Cost", min=0, max=float("inf"),
                           inclusive_min=True, inclusive_max=False)
input_schema.set_data_type("inflow", "Quantity", min=-float("inf"), max=float("inf"),
                           inclusive_min=False, inclusive_max=False)

# The default-default of zero makes sense everywhere except for Capacity
input_schema.set_default_value("arcs", "Capacity", float("inf"))
# ---------------------------------------------------------------------------------

# ------------------------ define the output schema -------------------------------
solution_schema = TicDatFactory(
        flow = [["Commodity", "Source", "Destination"], ["Quantity"]],
        parameters = [["Key"],["Value"]])
# ---------------------------------------------------------------------------------

# ------------------------ solving section-----------------------------------------
def solve(dat):
    """
    core solving routine
    :param dat: a good ticdat for the dataFactory
    :return: a good ticdat for the solutionFactory, or None
    """
    assert input_schema.good_tic_dat_object(dat)
    assert not input_schema.find_foreign_key_failures(dat)
    assert not input_schema.find_data_type_failures(dat)

    dat = input_schema.copy_to_pandas(dat, drop_pk_columns=False)

    # Create optimization model
    m = gu.Model('netflow')

    flow = Sloc.add_sloc(dat.cost.join(dat.arcs, on = ["Source", "Destination"],
                                       how = "inner", rsuffix="_arcs").
              apply(lambda r : m.addVar(ub=r.Capacity, obj=r.Cost,
                            name= 'flow_%s_%s_%s' % (r.Commodity, r.Source, r.Destination)),
                    axis=1, reduce=True))
    flow.name = "flow"

    dat.arcs.join(flow.groupby(level=["Source", "Destination"]).sum()).apply(
        lambda r : m.addConstr(r.flow <= r.Capacity,
                               'cap_%s_%s'%(r.Source, r.Destination)), axis =1)

    # for readability purposes using a dummy variable thats always zero
    zero = m.addVar(lb=0, ub=0, name = "forcedToZero")

    # there is a more pandonic way to do this group of constraints, but lets
    # demonstrate .sloc for those who think it might be more intuitive
    for h,j in sorted(set(dat.inflow[abs(dat.inflow.Quantity) > 0].index).union(
        flow.groupby(level=['Commodity','Source']).groups.keys(),
        flow.groupby(level=['Commodity','Destination']).groups.keys())):
            m.addConstr((gu.quicksum(flow.sloc[h,:,j]) or zero) +
                        dat.inflow.Quantity.loc[h,j] ==
                        (gu.quicksum(flow.sloc[h,j,:]) or zero),
                        'node_%s_%s' % (h, j))

    # Compute optimal solution
    m.optimize()

    if m.status == gu.GRB.status.OPTIMAL:
        t = flow.apply(lambda r : r.x)
        # TicDat is smart enough to handle a Series for a single data field table
        rtn = solution_schema.TicDat(flow = t[t > 0])
        rtn.parameters["Total Cost"] = dat.cost.join(t).apply(lambda r: r.Cost * r.flow,
                                                              axis=1).sum()
        return rtn
# ---------------------------------------------------------------------------------

# ------------------------ provide stand-alone functionality ----------------------
# when run from the command line, will read/write xls/csv/db/mdb files
if __name__ == "__main__":
    standard_main(input_schema, solution_schema, solve)
# ---------------------------------------------------------------------------------
