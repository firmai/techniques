from csv import DictReader
from os import path

from trader.TradeHandler import TradeHandler
from web3 import Web3

class TestTradeHandler:
    def test_get_data_from_fundamentals(self):
        trader = TradeHandler()
        trader.getData()
        data_path = path.join(
                path.dirname(path.realpath(__file__)),
                '..',
                'recommender',
                'data',
                'fundamentals'
        )
        with open(path.join(data_path, 'data.csv'), 'r') as fundamentals_csv:
            reader = DictReader(fundamentals_csv, ['ticker', 'indicator', 'dimension', 'date', 'value'])
            keys = ['ticker', 'indicator', 'dimension', 'date', 'value']
            checker = True
            for line in reader:
                for key in keys:
                    if not line[key]:
                        checker = False
                        break
        assert checker
