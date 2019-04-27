from netflow import input_schema
dat = input_schema.sql.create_tic_dat_from_sql("netflow_sample_data.sql")
input_schema.xls.write_file(dat, "Netflow_Sample_Data.xlsx", allow_overwrite=True)