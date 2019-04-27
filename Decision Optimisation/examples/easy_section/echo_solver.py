#!/usr/bin/python

# Copyright 2015, 2016, 2017 Opalytics, Inc.
#

# Simple toy solver to begin playing around with Python and ticdat.
# This solver does nothing but validate the input data passed sanity checks,
# and writes this same data back out as the 'solution'. This solver serves
# no purpose other than to provide an example for how to get started with
# ticdat.

# Provides command line interface via ticdat.standard_main
# For example, typing
#   python echo_solver.py -i input_data.xlsx -o input_copy_dir
# will read from a model stored in the file input_data.xlsx and write the same data back
# to .csv files in created directory input_copy_dir

from ticdat import TicDatFactory, standard_main

# ------------------------ define the input schema --------------------------------
# NOTE - defining the diet schema here.
# ***You should rewrite this section to define your own schema.***
# Please try to implement as much schema information as possible.

# There are three input tables, with 4 primary key fields and 4 data fields.
input_schema = TicDatFactory (
    categories = [["Name"],["Min Nutrition", "Max Nutrition"]],
    foods  = [["Name"],["Cost"]],
    nutrition_quantities = [["Food", "Category"], ["Quantity"]])

# Define the foreign key relationships
input_schema.add_foreign_key("nutrition_quantities", "foods", ["Food", "Name"])
input_schema.add_foreign_key("nutrition_quantities", "categories",
                            ["Category", "Name"])

# Define the data types
input_schema.set_data_type("categories", "Min Nutrition", min=0, max=float("inf"),
                           inclusive_min=True, inclusive_max=False)
input_schema.set_data_type("categories", "Max Nutrition", min=0, max=float("inf"),
                           inclusive_min=True, inclusive_max=True)
input_schema.set_data_type("foods", "Cost", min=0, max=float("inf"),
                           inclusive_min=True, inclusive_max=False)
input_schema.set_data_type("nutrition_quantities", "Quantity", min=0, max=float("inf"),
                           inclusive_min=True, inclusive_max=False)

# We also want to insure that Max Nutrition doesn't fall below Min Nutrition
input_schema.add_data_row_predicate(
    "categories", predicate_name="Min Max Check",
    predicate=lambda row : row["Max Nutrition"] >= row["Min Nutrition"])

# The default-default of zero makes sense everywhere except for Max Nutrition
input_schema.set_default_value("categories", "Max Nutrition", float("inf"))
# ---------------------------------------------------------------------------------


# ------------------------ define the output schema -------------------------------
# Since this solver does nothing other than echo the input data back out as a
# solution, the solution schema is the same as the input schema.
solution_schema = input_schema
# ---------------------------------------------------------------------------------


# ------------------------ create a solve function --------------------------------

def solve(dat):
    assert input_schema.good_tic_dat_object(dat)
    assert not input_schema.find_foreign_key_failures(dat)
    assert not input_schema.find_data_type_failures(dat)
    assert not input_schema.find_data_row_failures(dat)

    return dat  # i.e. echo
# ---------------------------------------------------------------------------------

# ------------------------ provide stand-alone functionality ----------------------
# when run from the command line, will read/write xls/csv/db/sql/mdb files
if __name__ == "__main__":
    standard_main(input_schema, solution_schema, solve)
# ---------------------------------------------------------------------------------
