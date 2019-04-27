var gasVal = 4712388;

contract('HedgeContract1', function(accounts) {
    before(function() {
        web3._extend({
            property: 'evm',
            methods: [new web3._extend.Method({
                name: 'snapshot',
                call: 'evm_snapshot',
                params: 0,
                outputFormatter: toIntVal
            })]
        });

        web3._extend({
            property: 'evm',
            methods: [new web3._extend.Method({
                name: 'revert',
                call: 'evm_revert',
                params: 1,
                inputFormatter: [toIntVal]
            })]
        });
    });

    it("increasing users", function() {
        var userScenarios = 20;

        // Fix investment
        var investment = web3.toWei(2, "ether");

        // Fix black list
        var blackList = [4,5,6];

        var hc = HedgeContract1.deployed();
        var initialState = web3.evm.snapshot();

        var userNoRange = Array.from(Array(userScenarios)).map((e, i) => i + 1);
        eachSeries(userNoRange, function(noOfUsers) {
            return investmentGasConsumption(noOfUsers, hc, accounts, investment, blackList, initialState).then(function(totalGas) {
                console.log(totalGas);
                return;
            });
        });
    });

    // it("changing blackList", function() {
    //     var blackListScenarios = 10;
    //
    //     // Fix investment
    //     var investment = web3.toWei(2, "ether");
    //
    //     // Fix no users
    //     var noOfUsers = 10;
    //
    //     var hc = HedgeContract1.deployed();
    //     var initialState = web3.evm.snapshot();
    //
    //     var blackListCompanies = [blackListScenarios];
    //     var max = 48;
    //     var min = 4;
    //     for (var i = 0; i < blackListScenarios; i++) {
    //         var str = [];
    //         for (var j = 0; j < i + 1; j++) {
    //             str.push(Math.floor(Math.random() * (max - min + 1)) + min);
    //         }
    //         blackListCompanies[i] = str;
    //     }
    //
    //     eachSeries(blackListCompanies, function(blackList) {
    //       // console.log(blackList);
    //         return investmentGasConsumption(noOfUsers, hc, accounts, investment, blackList, initialState).then(function(totalGas) {
    //             console.log(totalGas);
    //             return;
    //         });
    //     });
    // });
});

function investmentGasConsumption(maxUsers, hc, accounts, investment, blackList, initialState) {
    return new Promise((resolve, reject) => {
        var promises = [];
        var buyAgentAddress = accounts[maxUsers];
        var investAgentAddress = accounts[maxUsers];
        var investmentOffer = investment * maxUsers;

        promises.push(hc.setBuyAgent(buyAgentAddress));
        promises.push(hc.setInvestAgent(investAgentAddress));

        Promise.all(promises).then(function() {
            promises = [];
            var startBlockNumber = web3.eth.blockNumber;

            // Make investment
            for (var i = 0; i < maxUsers; i++) {
                promises.push(makeInvestment(hc, blackList, investment, gasVal, accounts[i]));
            }

            Promise.all(promises).then(function() {
                makeInvestmentStrategy(hc, [1,2,3], investmentOffer, gasVal, investAgentAddress) // Offer by invest agent
                    .then(withdrawBuyAgent(hc, gasVal, buyAgentAddress)) // Withdraw it - buy agent
                    .then(sendBuyAgent(hc, 1, 1, 1, 1, investmentOffer, gasVal, buyAgentAddress)) // Return investment - buy agent
                    .then(function() {
                        promises = [];

                        for (var i = 0; i < maxUsers; i++) {
                            promises.push(withdrawalUser(hc, gasVal, accounts[i])); // Withdrawal by users
                        }

                        Promise.all(promises).then(function() {
                            var totalGas = 0;
                            for (var i = startBlockNumber + 1; i <= web3.eth.blockNumber; i++) {
                                totalGas += outpusGas(i);
                            }
                            // gasBreakdown(startBlockNumber + 1, web3.eth.blockNumber, maxUsers);
                            web3.evm.revert(initialState);
                            resolve(maxUsers + ',' + totalGas + ',' + blackList.length);
                        });
                    })
                    .catch(function(err) {
                        console.log(err);
                    });
            });
        });
    });
}

function makeInvestment(contract, blacklist, investmentValue, gasValue, account) {
    return contract.createInvestment.sendTransaction(blacklist, {
        value: investmentValue,
        gas: gasValue,
        from: account
    });
}

function makeInvestmentStrategy(contract, blacklist, investmentOfferValue, gasValue, account) {
    return contract.investOffer.sendTransaction(investmentOfferValue, blacklist, {
        gas: gasValue,
        from: account
    });
}

function withdrawBuyAgent(contract, gasValue, account) {
    return contract.withdrawBuyAgent.sendTransaction({
        gas: gasValue,
        from: account
    });
}

function sendBuyAgent(contract, returnRatio, sharpe, alpha, beta, investmentReturnValue, gasValue, account) {
    return contract.sendBuyAgent.sendTransaction(returnRatio, sharpe, alpha, beta, {
        value: investmentReturnValue,
        gas: gasValue,
        from: account
    });
}

function withdrawalUser(contract, gasValue, account) {
    return contract.withdrawalUser.sendTransaction({
        gas: gasValue,
        from: account
    });
}

function outpusGas(blockNumber) {
    var transactions = web3.eth.getBlock(blockNumber).transactions;
    for (var i = 0; i < transactions.length; i++) {
        var transactionReceipt = web3.eth.getTransactionReceipt(transactions[i].hash);
        return transactionReceipt.gasUsed;
    }
}

function gasBreakdown(startBlock, endBlock, noUsers) {
    // One transaction one block in testrpc

    var output = [];
    var gas = 0;
    var blockIndex = startBlock;

    // Making investment
    for (var i = 0; i < noUsers; i++) {
        gas += outpusGas(blockIndex);
        blockIndex++;
    }
    output.push(gas);
    gas = 0;

    // Offer
    output.push(outpusGas(blockIndex));
    blockIndex++;

    // withdrawal - agent
    output.push(outpusGas(blockIndex));
    blockIndex++;

    // Execution and return value
    output.push(outpusGas(blockIndex));
    blockIndex++;

    // User withdrawal
    for (var i = 0; i < noUsers; i++) {
        gas += outpusGas(blockIndex);
        blockIndex++;
    }
    output.push(gas);
    var totalGas = output.reduce((a,b) => a + b, 0);

    console.log(noUsers + ',' + output.join() + ',' + totalGas);
}

function toIntVal(val) {
    return parseInt(val);
}

// define helper function that works kind of like async.eachSeries
function eachSeries(arr, iteratorFn) {
    return arr.reduce(function(p, item) {
        return p.then(function() {
            return iteratorFn(item);
        });
    }, Promise.resolve());
}
