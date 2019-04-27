# Inspired by https://www.quantopian.com/posts/grahamfundmantals-algo-simple-screening-on-benjamin-graham-number-fundamentals
#    Trading Strategy using Fundamental Data
#    1. Filter the top 50 companies by market cap
#    2. Find the top two sectors that have the highest average PE ratio
#    3. Every month exit all the positions before entering new ones at the month
#    4. Log the positions that we need
from csv import DictReader
from os import path, remove
from datetime import datetime
import numpy as np
import pandas as pd
import wget
import json


from zipline.api import (
    symbol,
    order_target_percent,
    record,
    schedule_function,
    date_rules,
    time_rules,
    )
from zipline.errors import SymbolNotFound

from contract.ContractHandler import ContractHandler


def initialize(context):
    # Ethereum contract
    #context.contract = ContractHandler()
    # Get blacklist of sectors which returns a list of codes
    #context.blacklist = context.contract.getBlacklist()
    # for testing purpose without Ethereum exclude Defense, Beer & Alcohol, Tabacco and Coal
    context.blacklist = [26.0, 4.0, 5.0, 29.0]

    # Only run get_fundamentals when necessary based on the rebalance function
    context.initial = True
    # Dictionary of stocks and their respective weights
    context.stock_weights = {}
    # Count of days before rebalancing
    context.days = 0
    # Number of sectors to go long in
    context.sect_numb = 2
    # Sector mappings
    context.sector_mappings = get_sectors(key='code')
    context.ticker_sector_dict = get_sector_code()

    # TODO: Update this accordingly (weekly?)
    # Rebalance monthly on the first day of the month at market open
    schedule_function(rebalance,
                      date_rule=date_rules.month_start(),
                      time_rule=time_rules.market_open())


def rebalance(context, data):
    # update the fundamentals data
    if not context.initial:
        get_fundamentals(context, data)

    # Exit all positions before starting new ones
    for stock in context.portfolio.positions:
        if stock not in context.fundamental_df:
            order_target_percent(stock, 0)

    print("The two sectors we are ordering today are %r" % context.sectors)

    # Create weights for each stock
    weight = create_weights(context, context.stocks)

    # Rebalance all stocks to target weights
    for stock in context.fundamental_df:
        if weight != 0:
            print("Ordering %0.0f%% percent of %s in %s" %
                  (weight * 100,
                   stock.symbol,
                   context.sector_mappings[context.fundamental_df[stock]['sector_code']]))
        order_target_percent(stock, weight)

    # track how many positions we're holding
    record(num_positions=len(context.fundamental_df))


def get_fundamentals(context, data):
    print("Updating fundamentals data")
    fundamentals = dict()
    with open(path.join('data', 'fundamentals', 'data.csv'), 'r') as fundamentals_csv:
        reader = DictReader(fundamentals_csv, ['ticker', 'indicator', 'dimension', 'date', 'value'])
        thisticker = ''

        values = dict()
        for line in reader:
            # print("Processing line {}".format(line))
            try:
                symbol_ticker = symbol(line['ticker'])
                if data.can_trade(symbol_ticker):
                    # Store most recent values in the ticker
                    if thisticker != symbol_ticker:
                        # print("Processing {}".format(symbol_ticker))
                        if not thisticker:
                            thisticker = symbol_ticker
                        else:
                            # add the sector code
                            try:
                                values['sector_code'] = context.ticker_sector_dict[thisticker.symbol]
                                if values['sector_code'] and values['pe_ratio'] and values['market_cap']:
                                    fundamentals[thisticker] = values
                                    # print("Adding {}".format(values))
                                values = dict()
                            except KeyError as e:
                                # print("Error on adding {}".format(e))
                                pass
                            thisticker = symbol_ticker

                    # Select only data that was available at that time
                    date = data.current(symbol_ticker, "last_traded").replace(tzinfo=None)
                    if date > datetime.strptime(line['date'], '%Y-%m-%d'):
                        # Set PE Ratio
                        if line['indicator'] in 'EPS':
                            values['pe_ratio'] = float(line['value'])
                        # Set Market Cap
                        elif line['indicator'] in 'SHARESWA':
                            price = data.current(symbol_ticker, "price")
                            totalshares = float(line['value'])
                            market_cap = price * totalshares
                            # Only consider stock with at least 10 million market cap
                            if market_cap > 10000000:
                                values['market_cap'] = price * totalshares
            except SymbolNotFound as e:
                pass
    # convert dict to DataFrame
    fundamentals_df = pd.DataFrame.from_dict(fundamentals)
    # Find sectors with the highest average PE
    sector_pe_dict = dict()
    for stock in fundamentals_df:
        try:
            sector = fundamentals_df[stock]['sector_code']
            pe = fundamentals_df[stock]['pe_ratio']

            # If it exists and is not included in the blacklist add our pe to the existing list.
            # Otherwise don't add it.
            if (sector not in sector_pe_dict) and (sector not in context.blacklist):
                sector_pe_dict[sector] = []

            sector_pe_dict[sector].append(pe)
        except KeyError as e:
            print("KeyError on {} at stock {}".format(e, stock))

    # Find average PE per sector
    sector_pe_dict = dict([
        (sectors, np.average(sector_pe_dict[sectors]))
        for sectors in sector_pe_dict
        if len(sector_pe_dict[sectors]) > 0
    ])

    # sector_pe_dict_avg = dict()
    # for sector in sector_pe_dict:
    #     if len(sector_pe_dict[sector]) > 0:
    #         sector_pe_dict_avg[sector] = np.mean(sector_pe_dict[sector], axis=0)

    # Sort in ascending order
    sectors = sorted(
            sector_pe_dict,
            key=lambda x: sector_pe_dict[x],
            reverse=True
    )[:context.sect_numb]

    # Filter out only stocks with that particular sector
    context.stocks = [
        stock for stock in fundamentals_df
        if fundamentals_df[stock]['sector_code'] in sectors
        ]

    # Initialize a context.sectors variable
    context.sectors = [context.sector_mappings[sect] for sect in sectors]

    # Update context.fundamental_df with the securities (and pe_ratio) that we need
    context.fundamental_df = fundamentals_df[context.stocks]

    # context.update_universe(context.fundamental_df.columns.values)


def before_trading_start(context, data):
    if context.initial:
        get_fundamentals(context, data)
        context.initial = False


def create_weights(context, stocks):
    """
        Takes in a list of securities and weights them all equally
    """
    if len(stocks) == 0:
        return 0
    else:
        weight = 1.0/len(stocks)
        return weight


def handle_data(context, data):
    """
      Code logic to run during the trading day.
      handle_data() gets called every bar.
    """
    pass


def get_sectors(key):
    # get sectors based on key (either code or description)
    sectors = dict()
    with open(path.join('data', 'fundamentals', 'Famacodes48.txt'), 'r') as sectorfile:
        for line in sectorfile:
            single_sector = line.split(',', maxsplit=1)
            if key in 'description':
                sectors[single_sector[1].rstrip()] = int(single_sector[0])
            elif key in 'code':
                sectors[int(single_sector[0])] = single_sector[1].rstrip()
    return sectors


def get_sector_code():
    sectors = get_sectors(key='description')
    sector_code = dict()
    url = "http://www.sharadar.com/meta/sf0-tickers.json"
    sf0tickers = wget.download(url)
    with open(sf0tickers) as file:
        data = json.load(file)
    for item in data:
        try:
            sector_code[str(item["Ticker"])] = sectors[item["Fama Industry"]]
        except KeyError as e:
            print('{} sector not found: {}'.format(item["Ticker"], e))
    remove(sf0tickers)
    return sector_code
