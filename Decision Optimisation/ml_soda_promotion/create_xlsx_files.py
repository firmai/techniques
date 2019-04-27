# Running this script will generate the .xlsx files and thus allow you to reproduce the
# results in beer_promotion.ipynb. Also generates optimizer_only_test_input.xlsx for exercising
# soda_promotion_optimizer.py and predict_then_optimize_super_bowl.xlsx for exercising the
# pipelined predictor app on the Opalytics Cloud Platform.

from ticdat import TicDatFactory
tdfHist = TicDatFactory(data =[[],
                    ['Product','Sales','Cost Per Unit','Easter Included','Super Bowl Included',
                     'Christmas Included', 'Other Holiday', '4 Wk Avg Temp',
                     '4 Wk Avg Humidity', 'Sales M-1 weeks', 'Sales M-2 weeks', 'Sales M-3 weeks',
                     'Sales M-4 Weeks', 'Sales M-5 weeks']])

histDat = tdfHist.json.create_tic_dat("soda_sales_historical_data.json")
tdfHist.xls.write_file(histDat, "soda_sales_historical_data.xlsx", allow_overwrite=True)

tdfSuperBowl = TicDatFactory(data = [[],
                    ['Product','Cost Per Unit','Easter Included','Super Bowl Included',
                     'Christmas Included', 'Other Holiday', '4 Wk Avg Temp', '4 Wk Avg Humidity',
                     'Sales M-1 weeks', 'Sales M-2 weeks', 'Sales M-3 weeks', 'Sales M-4 Weeks',
                     'Sales M-5 weeks']])

superBowlDat = tdfSuperBowl.json.create_tic_dat("super_bowl_promotion_data.json")
tdfSuperBowl.xls.write_file(superBowlDat, "super_bowl_promotion_data.xlsx", allow_overwrite=True)

import soda_promotion_optimizer as spo
spo_dat = spo.input_schema.sql.create_tic_dat_from_sql("optimizer_only_test_input.sql")
spo.input_schema.xls.write_file(spo_dat, "optimizer_only_test_input.xlsx", allow_overwrite=True)

sch1, sch2 = [_.schema() for _ in (tdfSuperBowl, spo.input_schema)]

tdfSuperBowl = TicDatFactory(forecast_sales = sch1["data"], **{k:v for k,v in sch2.items() if k != "forecast_sales"})
superBowlDat = tdfSuperBowl.TicDat(forecast_sales = superBowlDat.data,
                                   products  = {'11 Down': 'Clear', 'AB Root Beer': 'Dark',
                                                'Alpine Stream': 'Clear', 'Bright': 'Clear',
                                                'Crisp Clear': 'Clear', 'DC Kola': 'Dark',
                                                'Koala Kola': 'Dark', 'Mr. Popper': 'Dark',
                                                'Popsi Kola': 'Dark'},
                                   max_promotions = {"Clear":2, "Dark":2},
                                   parameters = {"Maximum Total Investment":750})
tdfSuperBowl.xls.write_file(superBowlDat, "predict_then_optimize_super_bowl.xlsx", allow_overwrite=True)

