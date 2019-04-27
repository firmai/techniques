"""
Trading Strategy using Fundamental Data
Adjusted Graham Fundamentals trading algorithm (based on https://www.quantopian.com/help#fundamental-data)
1. Filter the top 50 companies by market cap
2. Exclude companies based on ESC criteria
3. Find the top two sectors that have the highest average PE ratio
4. Every week exit all the positions before entering new ones
5. Log the positions that we need
"""
from os import path, environ

from zipline.api import (
    set_do_not_order_list,
    attach_pipeline,
    pipeline_output,
    order_target_percent,
    record,
    schedule_function,
    date_rules,
    time_rules,
    )
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing

import quandl

from recommender.FundamentalsHandler import PeRatio, MarketCap, SectorCode, SectorCodeFilter, SharesOutstanding
from contract.ContractHandler import ContractHandler


def initialize(context):
    # Ethereum contract
    context.contract = ContractHandler()

    # Set Quandl API key and import fundamentals data from quandl
    quandl.ApiConfig.api_key = environ['QUANDL_API_KEY']
    context.fundamentals_earnings = quandl.get_table('SF0', qopts={'columns': ['ticker']}).\
        to_numpy()

    # Get blacklist of companies which returns a list of SIDs
    blacklist = context.contract.getBlacklist()
    set_do_not_order_list(blacklist)

    # Dictionary of stocks and their respective weights
    context.stock_weights = {}

    # Count of days before rebalancing
    context.days = 0

    # Number of sectors to go long in
    context.sect_numb = 2

    # Sector mappings
    context.sector_mappings = get_sectors()

    # Rebalance weekly on the first day of the week at market open
    schedule_function(rebalance,
                      date_rule=date_rules.week_start(),
                      time_rule=time_rules.market_open())

    schedule_function(record_positions,
                      date_rule=date_rules.week_start(),
                      time_rule=time_rules.market_close())

    # Register pipeline
    fundamental_df = make_pipeline(context)
    attach_pipeline(fundamental_df, 'fundamentals')


#  Create a fundamentals data pipeline
def make_pipeline(context):

    # pe_ratio = close price / earnings per share
    pe_ratio = PeRatio([USEquityPricing.close, context.fundamentals_quandl])
    # mapping of companies to industry sectors
    sector_code = SectorCode()
    # market_cap = price * total shares outstanding (at e.g. closing)
    market_cap = MarketCap()
    # total shares outstanding reported by the company
    shares_outstanding = SharesOutstanding()
    # Filter two most profitable sectors
    sector_filter = SectorCodeFilter()

    fundamentals = Pipeline(
        columns={
            'sector_code': sector_code,
            'pe_ratio': pe_ratio,
            'market_cap': market_cap,
            'shares_outstanding': shares_outstanding,
        },
        screen={
            sector_filter,
            market_cap,
            shares_outstanding is not None,
        }
    )
    return fundamentals


"""
Called before the start of each trading day.  Runs the
fundamentals query and saves the matching securities into
context.fundamentals_df.
"""


def before_trading_start(context, data):
    # May need to increase the number of stocks
    num_stocks = 50
    context.fundamental_df = pipeline_output('fundamentals')
    # Get blacklist of companies which returns a list of SIDs
    blacklist = context.contract.getBlacklist()
    set_do_not_order_list(blacklist)
    # get current balance to determine capital_base
    context.capital_base = context.contract.getBalance()
    # update quandl fundamentals
    context.fundamentals_quandl = quandl.Datatable('SF0').data().to_numpy()


def create_weights(stocks):
    # Takes in a list of securities and weights them all equally
    if len(stocks) == 0:
        return 0
    else:
        weight = 1.0 / len(stocks)
        return weight


def rebalance(context, data):
    # Exit all positions before starting new ones
    for stock in context.portfolio.positions:
        if stock not in context.fundamental_df and data.can_trade(stock):
            order_target_percent(stock, 0)

    # Create weights for each stock
    weight = create_weights(context.stocks)

    # Rebalance all stocks to target weights
    for stock in context.fundamental_df:
        if data.can_trade(stock):
            if weight != 0:
                code = context.sector_mappings[
                     context.fundamental_df[stock]['sector_code']
                ]

            order_target_percent(stock, weight)


def record_positions(context, data):
    # track how many positions we're holding
    record(num_positions=len(context.portfolio.positions))


def get_sectors():
    sectors = dict()
    with open(path.join('data', 'fundamentals', 'Famacodes48.txt'), 'r') as sectorfile:
        for line in sectorfile:
            single_sector = line.split(',', maxsplit=1)
            sectors[int(single_sector[0])] = single_sector[1].rstrip()
    return sectors
