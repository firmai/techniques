from diet import input_schema
dat = input_schema.sql.create_tic_dat_from_sql("diet_sample_data.sql")
# writing out an .xls file instead of an .xlsx file because the .xlsx writer
# doesn't handle infinty seamlessly
input_schema.xls.write_file(dat, "Diet_Sample_Data.xls", allow_overwrite=True)