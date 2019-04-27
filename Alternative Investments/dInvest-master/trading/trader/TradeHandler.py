# TODO: How to execute the same strategy twice in case new strategies are rejected?
from csv import DictReader, DictWriter
from datetime import datetime, timedelta
import schedule
from subprocess import call
from os import path, environ, remove
from time import sleep
import wget
import zipfile


class TradeHandler:
    def __init__(self):
        # for testing
        self.startdate = "2016-10-20"
        # startdate = (datetime.today() - timedelta(days=2)).strftime('%Y-%m-%d')
        self.algorithm = path.join(
                path.join(path.dirname(path.realpath(__file__)), '..'),
                'trader',
                'buyapple.py'
        )
        self.resultpath = path.join(
                path.join(path.dirname(path.realpath(__file__)), '..'),
                'analysis',
                'results',
        )
        self.quandl_api_key = environ['QUANDL_API_KEY']

    def getTrader(self):
        enddate = self.getCurrentDate()
        resultfile = path.join(self.resultpath, 'value{}.pickle'.format(enddate))
        # TODO: rewrite to function https://groups.google.com/forum/#!topic/zipline/FRF-hwTs2qM
        command = 'zipline run -f {} --start {} --end {} -o {}'.\
            format(self.algorithm, self.startdate, enddate, resultfile)
        call(command, shell=True)

    def getData(self):
        data_path = path.join(
                path.dirname(path.realpath(__file__)),
                '..',
                'recommender',
                'data',
                'fundamentals'
        )
        # Remove previous data
        if path.isfile(path.join(data_path, 'data.csv')):
            remove(path.join(data_path, 'data.csv'))
        # Download latest full fundamentals dataset
        url = 'https://www.quandl.com/api/v3/databases/SF0/data?auth_token={}'.format(self.quandl_api_key)
        fundamentals_zip = wget.download(url)
        # Extract file with format SF0_YYYYMMDD.csv
        with zipfile.ZipFile(fundamentals_zip, 'r') as zip_ref:
            zip_ref.extractall(path=data_path)
        remove(fundamentals_zip)
        # Cleanup data
        # today = datetime.today().strftime('%Y%m%d')
        today = '20161118'
        with open(path.join(data_path, 'SF0_{}.csv'.format(today)), 'r') as fundamentals_in, \
                open(path.join(data_path, 'data.csv'), 'w') as fundamentals_out:
            reader = DictReader(fundamentals_in, ['ticker_indicator_dimension', 'date', 'value'])
            writer = DictWriter(fundamentals_out, ['ticker', 'indicator', 'dimension', 'date', 'value'])
            for line in reader:
                ticker, indicator, dimension = line['ticker_indicator_dimension'].split('_')
                # Select only data that was available at that time
                if (indicator in 'EPS' or indicator in 'SHARESWA') and float(line['value']) != 0:
                    newline = {
                        'ticker': ticker,
                        'indicator': indicator,
                        'dimension': dimension,
                        'date': line['date'],
                        'value': line['value']
                    }
                    writer.writerow(newline)
        remove(path.join(data_path, 'SF0_{}.csv'.format(today)))

    def executeTrader(self):
        schedule.every().day.at("6:00").do(self.getData)
        schedule.every().day.at("6:30").do(self.getTrader)
        try:
            while True:
                print(schedule.jobs)
                schedule.run_pending()
                sleep(3600)
        except:
            print("An error occured on {}".format(datetime.today() - timedelta(days=1)))

    def getCurrentDate(self):
        yesterday = datetime.today() - timedelta(days=1)
        return yesterday.strftime('%Y-%m-%d')

    def getMetrics(self):
        # get risk performance metrics from latest trading
        # send risk performance metrics to contract
        return True

    def getEther(self):
        # get profit/loss from current trading day
        # add/substract from trading account
        # send Ether back to contract
        return True


def main():
    trader = TradeHandler()
    trader.executeTrader()
