from os import path
from csv import DictReader
import numpy as np

from zipline.pipeline.factors import CustomFactor
from zipline.pipeline.filters import CustomFilter
from zipline.pipeline.data import USEquityPricing
from datetime import datetime


def getFundamentals():
    today = datetime.today().strftime('%Y%m%d')
    fundamentals = dict()
    with open(path.join('data', 'fundamentals','SF0_{}.csv'.format(today)), 'r') as fundamentals_csv:
        reader = DictReader(fundamentals_csv, 'ticker_indicator_dimension,date,value')
        for line in reader:
            value = {'pe_ratio': {}, 'market_cap': {}}
            name, indicator = line['ticker_indicator_dimension'].split('_', maxsplit=1)
            if 'EPS_MRY' in indicator and float(line['value']) != 0:
                value['pe_ratio'] = {line['date']: float(line['value'])}
                fundamentals[name].append(value)

    return fundamentals

class PeRatio(CustomFactor):
    window_length = 1
    def compute(self, today, assets, out, closing_price, earnings):
        earnings = 0
        pe_ratio = closing_price / earnings
        out[:] = pe_ratio


class MarketCap(CustomFactor):
    def compute(self, today, assets, out, data):
        print('')


class SectorCode(CustomFactor):
    def compute(self, today, assets, out, *inputs):
        print('')


class SectorCodeFilter(CustomFilter):
    def compute(self, today, assets, out, *inputs):
        print('')


class SharesOutstanding(CustomFactor):
    def compute(self, today, assets, out, *inputs):
        print('')
