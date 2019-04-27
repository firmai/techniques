#Analysis of Hedge fund operational costs in Ethereum

This section will investigate how much gas it will cost to the platform to keep running the hedge fund. This is a very primitive analysis. Currently, all experiments are run locally instead of the actual Ethereum blockchain for convenience. However, since EVM (Ethereum Virtual Machine) is a Turing machine, the number of computations will be the same as the real blockchain, the figures computed is comparable with real blockchain.


## Experiment design
All tests were conducted using [testrpc](https://github.com/ethereumjs/testrpc) platform which is able to simulate full client behaviour of an Ethereum node locally. Furthermore, [truffle](https://github.com/ConsenSys/truffle) framework was used to develop the dInvest platform. It supports the Mocha testing framework for automated testing. All the experiments are expressed as tests in `truffle` and executed in `testrpc` node.

In a single run of the experiment,
1. All users will make the investment to the hedge fund (including the criteria)
2. Invest agent will make a fake strategy with a return of 20% of the hedge fund value that does not violate any criteria
3. Execution engine will accept the strategy and receive ether from hedge fund to execute it
4. Execution engine will send the returned investment to the hedge fund with financial indicators
5. All users will withdraw the full return of the investment.

## Consumption of gas w.r.t. number of users in the hedge fund
In this experiment, investment, user criteria (blacklist of companies) are fixed and the number of users will be changed 1 to 10. Thus, 10 runs of the hedge fund will be executed.

![](/home/tharidu/workspace/Dinvest/solidity/analysis/users.png) 

## Consumption of gas w.r.t. number of criteria in the hedge fund
In this experiment, investment, number of users are fixed and the number of criteria will be changed 1 to 10. Therefore, a minimum of 1 blacklisted company and a maximum of 10 blacklisted companies will be used for the users' investments.

![](/home/tharidu/workspace/Dinvest/solidity/analysis/blacklist.png) 

## Analysis

According to above figures, it is clear that with the increase of users or criteria the number of computations in EVM will exponentially increase. The reason for this abnormal usage of gas is because our contract functions are still not optimized. In addition to that, due to the separation of buy agent and invest agent, unneccesary interactions are happening, thus increasing the gas cost. We will be analyzing the individual contract functions to find out the exact places that needed to be optimized.

## Ether cost

The unit price of gas in testrpc is 1 Wei. However in the current Ethereum blockchain 1 gas costs 27306724366 Wei which is 0.000000027306724366 Ether (about 0.000000296 in USD - 1 Ether costs 10.84 USD as of 16/11/02).

According to current gas consumption in the hedge fund per investment (current worst case scenarios and after adjusting to correct gas prices),
10 users (with 3 user criteria) - 0.14 Ether (1.60 USD)
10 critera (with 5 users) - 0.13 Ether (1.48 USD)
