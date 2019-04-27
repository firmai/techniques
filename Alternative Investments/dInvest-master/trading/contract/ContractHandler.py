# TODO: Needs to listen to events of the organization
# Event: NewInvestmentByUser (address, value)
# TODO: Query list (blacklist) of companies from contract (one blacklist for all)
# TODO: Export financial indicators (return, Sharpe, alpha, beta etc.)
# TODO: Find financial offer and call function in contract
from web3 import Web3, RPCProvider
from os import path
import requests
import json


class ContractHandler:
    def __init__(self):
        # Load contract configuration
        self.web3 = Web3(RPCProvider(host='localhost', port='8545'))
        dir_path = path.dirname(path.realpath(__file__))
        with open(str(path.join(dir_path, 'Configuration.txt')), 'r') as configuration:
            for line in configuration:
                if line.startswith('contract='):
                    self.contract_address = line.split('=')[1].rstrip('\n')
                if line.startswith('trading_account='):
                    self.trading_account = line.split('=')[1].rstrip('\n')
                if line.startswith('trading_password='):
                    self.trading_password = line.split('=')[1].rstrip('\n')
                # Simulate trading by having a mining account to transfer ether
                if line.startswith('mining_account='):
                    self.mining_account = line.split('=')[1].rstrip('\n')
                if line.startswith('mining_password='):
                    self.mining_password = line.split('=')[1].rstrip('\n')
        with open(str(path.join(dir_path, 'contract_abi.json')), 'r') as abi_definition:
            self.abi = json.load(abi_definition)
        self.contract = self.web3.eth.contract(self.abi, self.contract_address)

    def setCurrentBalance(self, trading_result):
        # trading result represents the profit/loss as a float, e.g. 20% profit
        # would be represented by the float 0.2
        current_balance = self.getCurrentBalance()
        new_balance = current_balance * (1 + trading_result)
        transaction = dict()
        if new_balance > current_balance:
            transaction['from'] = self.mining_account
            transaction['to'] = self.trading_account
            transaction['value'] = new_balance
            self.web3.personal.unlockAccount(self.mining_account, self.mining_password)
        elif new_balance < current_balance:
            transaction['from'] = self.trading_account
            transaction['to'] = self.mining_account
            transaction['value'] = new_balance
            self.web3.personal.unlockAccount(self.trading_account, self.trading_password)
        if transaction:
            transaction_id = self.web3.eth.sendTransaction(transaction)
            return 'Updated balance with transaction {}'.format(transaction_id)

    def getCurrentBalance(self):
        r = requests.get('https://coinmarketcap-nexuist.rhcloud.com/api/eth')
        c = r.json()
        conversion = c.get('price').get('usd')
        wei_balance = self.web3.eth.getBalance(self.trading_account)
        ether_balance = Web3.fromWei(wei_balance, unit='ether')
        return conversion * float(ether_balance)

    def getBlacklist(self):
        # call contract function to get blacklist
        blacklist = self.contract.call().blackListCompanies()
        return blacklist

    def setInvestmentOffer(self, amount, companies):
        # Send investment offer and receive ether to investment account
        companies_str = str(companies)
        transaction_id = self.contract.call().investOffer(amount, companies_str)
        if transaction_id:
            self.contract.call().withdrawBuyAgent()
            return transaction_id
        return False

    def setMetrics(self, metrics):
        transaction = dict()
        transaction['value'] = self.getCurrentBalance()
        transaction['from'] = self.trading_account
        # sendBuyAgent metrics as int multiplied by 1000 as Ethereum cannot handle decimals
        sendmetrics = self.contract.call().sendBuyAgent(
            int(metrics['return']*1000),
            int(metrics['sharpe']*1000),
            int(metrics['alpha']*1000),
            int(metrics['beta']*1000),
            transaction
        )
        return sendmetrics
