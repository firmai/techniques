#!/usr/bin/python

# Copyright 2015, 2016 Opalytics, Inc.
#

# Implement core functionality needed to achieve modularity.
# 1. Define the input data schema
# 2. Define the output data schema
# 3. Create a solve function that accepts a data set consistent with the input
#    schema and (if possible) returns a data set consistent with the output schema.
#
# Provides command line interface via ticdat.standard_main
# For example, typing
#   python diet.py -i input_data.xlsx -o solution_data.xlsx
# will read from a model stored in the file input_data.xlsx and write the solution
# to solution_data.xlsx.
#
# Note that file requires diet.lng to be in the same directory

from ticdat import TicDatFactory, standard_main, lingo_run

# ------------------------ define the input schema --------------------------------
# There are three input tables, with 4 primary key fields and 4 data fields.
input_schema = TicDatFactory (
    categories = [["Name"],["Min Nutrition", "Max Nutrition"]],
    foods  = [["Name"],["Cost"]],
    nutrition_quantities = [["Food", "Category"], ["Quantity"]])

# add foreign key constraints
input_schema.add_foreign_key("nutrition_quantities", "foods", ['Food', 'Name'])
input_schema.add_foreign_key("nutrition_quantities", "categories", ['Category', 'Name'])

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
# There are three solution tables, with 2 primary key fields and 3 data fields.
solution_schema = TicDatFactory(
    parameters = [["Key"],["Value"]],
    buy_food = [["Food"],["Quantity"]],
    consume_nutrition = [["Category"],["Quantity"]])
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

    # These are the variables populated by the diet.lng file.
    solution_variables = TicDatFactory(buy_food=[["Food"],["Quantity"]])

    sln = lingo_run("diet.lng", input_schema, dat, solution_variables)
    if sln:
        rtn = solution_schema.TicDat(buy_food={k:r for k,r in sln.buy_food.items() if r["Quantity"] > 0})
        for (f, c), r in dat.nutrition_quantities.items():
            if f in rtn.buy_food:
                rtn.consume_nutrition[c]["Quantity"] += r["Quantity"] * rtn.buy_food[f]["Quantity"]
        rtn.parameters['Total Cost'] = sum(dat.foods[f]["Cost"] * r["Quantity"]
                                           for f,r in rtn.buy_food.items())
        return rtn
# ---------------------------------------------------------------------------------

# ------------------------ provide stand-alone functionality ----------------------
# when run from the command line, will read/write xls/csv/db/sql/mdb files
if __name__ == "__main__":
    standard_main(input_schema, solution_schema, solve)
# ---------------------------------------------------------------------------------