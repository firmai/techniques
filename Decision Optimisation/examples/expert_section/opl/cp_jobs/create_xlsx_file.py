from jobs import input_schema
dat = input_schema.sql.create_tic_dat_from_sql("jobs_sample_data.sql")
input_schema.xls.write_file(dat, "Jobs_Sample_Data.xlsx", allow_overwrite=True)