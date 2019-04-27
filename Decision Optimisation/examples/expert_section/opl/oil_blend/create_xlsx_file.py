from oil_blend import input_schema
dat = input_schema.json.create_tic_dat("oil_blend_sample_data.json")
input_schema.xls.write_file(dat, "Oil_Blend_Sample_Data.xlsx", allow_overwrite=True)