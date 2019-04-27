#!/usr/bin/python

# Copyright 2015, Opalytics, Inc.
#
# edited with permission from Gurobi Optimization, Inc.

# Solve a multi-commodity flow problem as python package.
# This version of the file uses pandas both for table and for complex data
# manipulation. There is no slicing or iterating over indexes.

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
from ticdat import TicDatFactory, standard_main

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

    flow = dat.cost.join(dat.arcs, on = ["Source", "Destination"],
                         how = "inner", rsuffix="_arcs")\
           .apply(lambda r : m.addVar(ub=r.Capacity, obj=r.Cost,
                                   name='flow_%s_%s_%s'%
                                        (r.Commodity, r.Source, r.Destination)),
                   axis=1, reduce=True)
    flow.name = "flow"

    # combining aggregate with gurobipy.quicksum is more efficient than using sum
    flow.groupby(level=["Source", "Destination"])\
        .aggregate({"flow": gu.quicksum})\
        .join(dat.arcs)\
        .apply(lambda r : m.addConstr(r.flow <= r.Capacity,
                                      'cap_%s_%s' %(r.Source, r.Destination)),
               axis =1)

    def flow_subtotal(node_fld, sum_field_name):
        rtn = flow.groupby(level=['Commodity',node_fld])\
                  .aggregate({sum_field_name : gu.quicksum})
        rtn.index.names = ['Commodity', 'Node']
        return rtn

    # We need a proxy for zero because of the toehold problem, and
    # we use quicksum([]) instead of a dummy variable because of the fillna problem.
    # (see notebooks in this directory and parent directory)
    zero_proxy = gu.quicksum([])
    flow_subtotal("Destination", "flow_in")\
        .join(dat.inflow[abs(dat.inflow.Quantity) > 0].Quantity, how="outer")\
        .join(flow_subtotal("Source", "flow_out"), how = "outer")\
        .fillna(zero_proxy)\
        .apply(lambda r : m.addConstr(r.flow_in + r.Quantity  - r.flow_out == 0,
                                      'cons_flow_%s_%s' % r.name),
               axis =1)

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
