# from contract.ContractHandler import ContractHandler
# from web3 import Web3
#
# class TestContractHandler:
#     def test_init_configuration(self):
#         contract = ContractHandler()
#         assert (contract.web3 and
#                 contract.contract_address and
#                 contract.trading_account and
#                 contract.password and
#                 contract.abi and
#                 contract.contract)
#
#     def test_get_usd_from_wei(self):
#         contract = ContractHandler()
#         wei = 1322132169
#         ether_balance = Web3.fromWei(wei, unit='ether')
#         usd_lower = 10 * float(ether_balance)
#         usd_upper = 15 * float(ether_balance)
#         assert usd_upper > contract.getUSDfromWei(wei) > usd_lower
#
#     def test_get_blacklist(self):
#         contract = ContractHandler()
#         blacklist = contract.getBlacklist()
#         assert blacklist is ''
#
#     def test_set_investment_offer(self):
#         contract = ContractHandler()
#         ether = 1
#         campanies_str = 'apple,intel,google'
#         transaction_id = contract.setInvestmentOffer(ether, campanies_str)
#         assert transaction_id == 1
#
#     def test_set_metrics(self):
#         metrics = {
#             'return': 20,
#             'sharpe': 65,
#             'alpha': 12,
#             'beta': 31
#         }
#         contract = ContractHandler()
#         assert contract.setMetrics(metrics)
