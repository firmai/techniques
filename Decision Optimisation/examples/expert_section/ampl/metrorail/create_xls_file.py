# Use this file to convert the metrorail_sample_data.json data set
# to Excel format.
#
#    python create_xls_file.py
#
# will create a file named Metro Rail Data.xlsx.
#
# metrorail.py will produce the same result regardless of whether
# it is run on metrorail_sample_data.json or Metro Rail Data.xlsx.
# This result is largely consistent with the heat map result from
# https://orbythebeach.wordpress.com/2018/03/01/buying-metrorail-tickets-in-miami/
# with the exception that we find only two infeasible sub-models.

from metrorail import input_schema
dat = input_schema.json.create_tic_dat("metrorail_sample_data.json")
input_schema.xls.write_file(dat, "Metro Rail Data.xlsx", allow_overwrite=True,
                            case_space_sheet_names=True)