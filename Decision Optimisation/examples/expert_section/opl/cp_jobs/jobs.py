#!/usr/bin/python

# Copyright 2015, 2016, 2017 Opalytics, Inc.
#
# Constraint programming example based on https://ibm.co/2rGVyet

# Implement core functionality needed to achieve modularity.
# 1. Define the input data schema
# 2. Define the output data schema
# 3. Create a solve function that accepts a data set consistent with the input
#    schema and (if possible) returns a data set consistent with the output schema.
#
# Provides command line interface via ticdat.standard_main
# For example, typing
#   python jobs.py -i input_data.xlsx -o solution_data.xlsx
# will read from a model stored in the file input_data.xlsx and write the solution
# to solution_data.xlsx.
#
# Note that file requires jobs.mod to be in the same directory

from ticdat import TicDatFactory, standard_main, opl_run
import time
import datetime

# ------------------------ define the input schema --------------------------------
# There are three input tables, with 4 primary key fields and 4 data fields.
input_schema = TicDatFactory (
    parameters = [["Key"],["Value"]],
    machines = [["Name"],[]],
    jobs = [["Name"],["Machine1","Durations1","Machine2","Durations2"]]
    )

input_schema.set_data_type("parameters", "Key", number_allowed=False, strings_allowed=("Load Duration",))
input_schema.set_data_type("parameters", "Value", min=0, max=float("inf"), inclusive_max=False)

input_schema.add_foreign_key("jobs", "machines", ["Machine1", "Name"])
input_schema.add_foreign_key("jobs", "machines", ["Machine2", "Name"])

input_schema.set_data_type("jobs", "Durations1",  min=0, max=float("inf"), inclusive_max=False, must_be_int=True)
input_schema.set_data_type("jobs", "Durations2",  min=0, max=float("inf"), inclusive_max=False, must_be_int=True)

input_schema.set_data_type("jobs", "Machine1", number_allowed=False, strings_allowed='*')
input_schema.set_data_type("jobs", "Machine2", number_allowed=False, strings_allowed='*')

input_schema.opl_prepend = "inp_" # avoid table name collisions
# ---------------------------------------------------------------------------------


# ------------------------ define the output schema -------------------------------
solution_schema = TicDatFactory(
    act = [["Job", "Task"],["Start", "End"]],
    act_as_dates = [["Job", "Task"],["Start Date", "End Date"]])

solution_schema.opl_prepend = "sln_"
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

    rtn = opl_run("jobs.mod", input_schema, dat, solution_schema)
    if rtn:
        # for simplicities sake, I'm just going to treat OPL time as 'minutes from now'
        now = time.time()
        format_iso = lambda x : datetime.datetime.utcfromtimestamp(now + x*60).isoformat() + "Z"
        for k,r in rtn.act.items():
            rtn.act_as_dates[k] = [format_iso(r["Start"]), format_iso(r["End"])]
        return rtn
# ---------------------------------------------------------------------------------

# ------------------------ provide stand-alone functionality ----------------------
# when run from the command line, will read/write json/xls/csv/db/sql/mdb files
if __name__ == "__main__":
    standard_main(input_schema, solution_schema, solve)
# ---------------------------------------------------------------------------------
