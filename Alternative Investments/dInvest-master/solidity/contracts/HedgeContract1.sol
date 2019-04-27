pragma solidity ^0.4.2;

contract HedgeContract1 {
  struct Investment {
    address investor;
    uint value;
    /*uint nowValue;*/
    /*uint holding;*/
    uint withdrawal;
    uint period; // in days
    uint withdrawalLimit;
    uint8[] sectors; //TODO limit size
  }

  address public creator;
  address public investAgent;
  address public buyAgent;
  uint public minimumInvestment;
  uint public originalInvestment;

  // Performance indicators
  int public returnRatio;
  int public sharpe;
  int public alpha;
  int public beta;
  // TODO - Getters and setters

  Investment[] public investments;
  mapping (address => uint) pendingWithdrawals;
  mapping (uint8 => uint8) blackList;

  // Events - publicize actions to external listeners
  event InvestmentOfferByBot(uint amount);
  event NewInvestmentByUser(address accountAddress, uint amount);

  // Helper function to guard functions
  modifier onlyBy(address _account)
  {
    if (msg.sender != _account)
        throw;
    _;
  }

  function HedgeContract1(
      uint _minimumInvestment,
      address _investAgent,
      address _buyAgent
    ) {
      // Set contract creator when creating the contract
      creator = msg.sender;
  }

  // Set new invest agent - only owner
  function setInvestAgent(address newInvestAgent)
    onlyBy(creator)
  {
    investAgent = newInvestAgent;
  }

  // Set new buy agent - only owner
  function setBuyAgent(address newBuyAgent)
    onlyBy(creator)
  {
    buyAgent = newBuyAgent;
  }

  // Set new minimum investment - only owner
  function setMinimumInvestment(uint newMinimumInvestment)
    onlyBy(creator)
  {
    // TODO - If its the same do not update to save gas
    minimumInvestment = newMinimumInvestment;
  }

  // Create a new investment
  function createInvestment(uint8[] sectorList) payable {
    if (msg.value < minimumInvestment) {
      throw;
    }

    // TODO - existing investment?
    // 3, 1 to change
    investments.push(Investment(msg.sender, msg.value, 0, 3, 1, sectorList));

    for(uint x = 0; x < sectorList.length; x++) {
        blackList[sectorList[x]] = sectorList[x];
    }

    // Publish event
    NewInvestmentByUser(msg.sender, msg.value);
  }

  // Investment opportunity - only agent
  function investOffer(uint amount, uint8[] sectorList)
    onlyBy(investAgent)
  {
    if (this.balance < amount) {
      throw;
    }

    // TODO - implement criteria here
    bool criteria = blackListSectorExists(sectorList);

    if (!criteria) {
      pendingWithdrawals[buyAgent] += amount;
      originalInvestment = amount;
      InvestmentOfferByBot(amount); // fire event
    } else {
      throw;
    }
  }

  function withdrawBuyAgent() onlyBy(buyAgent)
    returns (bool)
  {
        uint amount = pendingWithdrawals[msg.sender];

        pendingWithdrawals[msg.sender] = 0;
        if (msg.sender.send(amount)) {
            return true;
        } else {
            pendingWithdrawals[msg.sender] = amount;
            return false;
        }
  }

  function sendBuyAgent(int _returnRatio, int _sharpe, int _alpha, int _beta) payable
    onlyBy(buyAgent)
  {
    // Receive ether
    // Set financial indicators
    returnRatio = _returnRatio;
    sharpe = _sharpe;
    alpha = _alpha;
    beta = _beta;

    // Contract need to divide ether according to the share
    for(uint x = 0; x < investments.length; x++) {
      investments[x].withdrawal = investments[x].value * msg.value / originalInvestment; // TODO - Check truncations
      investments[x].value = 0;
    }

    clearBlacklistMapping();
  }

  function clearBlacklistMapping() internal constant
  {
    for(uint8 x = 0; x < 48; x++) {
      blackList[x] = 0;
    }
  }

  function blackListSectorExists(uint8[] sectors) internal constant returns (bool)
  {
    for(uint x = 0; x < sectors.length; x++) {
      if(blackList[sectors[x]] != address(0x0) || blackList[sectors[x]] != 0) {
          return true;
      }
    }

    return false;
  }

  function blackListCompanies() constant returns (uint8[48])
  {
    uint8[48] memory arr;
    for(uint x = 0; x < investments.length; x++) {
        for(uint y = 0; y < investments[x].sectors.length; y++) {
            arr[investments[x].sectors[y] - 1] = investments[x].sectors[y];
        }
    }

    return arr;
  }

  function availableForInvestment() constant returns (uint)
  {
    uint availableForInvestment;

    for(uint x = 0; x < investments.length; x++) {
      availableForInvestment += investments[x].value;
    }

    return availableForInvestment;
  }

  // Withdrawal by user
  function withdrawalUser() public returns (bool)
  {
    for(uint x = 0; x < investments.length; x++) {
      if (msg.sender == investments[x].investor) {
        uint amount = investments[x].withdrawal;

        investments[x].withdrawal = 0;
        if (msg.sender.send(amount)) {
            return true;
        } else {
            investments[x].withdrawal = amount;
            return false;
        }
        break;
      }
    }

    return false;
  }

  // Kill the contract and send the funds to creator
  // TODO - This has to be improved - send funds according to nowValue back to investors
  function kill() {
    if (msg.sender == creator) suicide(creator);
  }

  // TODO - get the values from the investments array
  // Get investment details from investments mapping
  function getInvestmentCurrentValue(address investor) constant
    returns(uint nowValue)
  {
    // nowValue = investments[investor].nowValue;
  }
}
