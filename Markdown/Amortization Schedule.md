

```python
import pandas as pd
from datetime import date
import numpy as np
from collections import OrderedDict
from dateutil.relativedelta import *
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
```

Build a payment schedule using a generator that can be easily read into a pandas dataframe for additional analysis and plotting


```python
def amortize(principal, interest_rate, years, pmt, addl_principal, start_date, annual_payments):
    """
    Calculate the amortization schedule given the loan details.

    :param principal: Amount borrowed
    :param interest_rate: The annual interest rate for this loan
    :param years: Number of years for the loan
    :param pmt: Payment amount per period
    :param addl_principal: Additional payments to be made each period.
    :param start_date: Start date for the loan.
    :param annual_payments: Number of payments in a year.

    :return: 
        schedule: Amortization schedule as an Ortdered Dictionary
    """

    # initialize the variables to keep track of the periods and running balances
    p = 1
    beg_balance = principal
    end_balance = principal
    
    while end_balance > 0:
        
        # Recalculate the interest based on the current balance
        interest = round(((interest_rate/annual_payments) * beg_balance), 2)
        
        # Determine payment based on whether or not this period will pay off the loan
        pmt = min(pmt, beg_balance + interest)
        principal = pmt - interest
        
        # Ensure additional payment gets adjusted if the loan is being paid off
        addl_principal = min(addl_principal, beg_balance - principal)
        end_balance = beg_balance - (principal + addl_principal)

        yield OrderedDict([('Month',start_date),
                           ('Period', p),
                           ('Begin Balance', beg_balance),
                           ('Payment', pmt),
                           ('Principal', principal),
                           ('Interest', interest),
                           ('Additional_Payment', addl_principal),
                           ('End Balance', end_balance)])
        
        # Increment the counter, balance and date
        p += 1
        start_date += relativedelta(months=1)
        beg_balance = end_balance
```

Wrapper function to call `amortize`.

This function primarily cleans up the table and provides summary stats so it is easy to compare various scenarios.


```python
def amortization_table(principal, interest_rate, years,
                       addl_principal=0, annual_payments=12, start_date=date.today()):
    """
    Calculate the amortization schedule given the loan details as well as summary stats for the loan

    :param principal: Amount borrowed
    :param interest_rate: The annual interest rate for this loan
    :param years: Number of years for the loan
    
    :param annual_payments (optional): Number of payments in a year. DEfault 12.
    :param addl_principal (optional): Additional payments to be made each period. Default 0.
    :param start_date (optional): Start date. Default first of next month if none provided

    :return: 
        schedule: Amortization schedule as a pandas dataframe
        summary: Pandas dataframe that summarizes the payoff information
    """
    
    # Payment stays constant based on the original terms of the loan
    payment = -round(np.pmt(interest_rate/annual_payments, years*annual_payments, principal), 2)
    
    # Generate the schedule and order the resulting columns for convenience
    schedule = pd.DataFrame(amortize(principal, interest_rate, years, payment,
                                     addl_principal, start_date, annual_payments))
    schedule = schedule[["Period", "Month", "Begin Balance", "Payment", "Interest", 
                         "Principal", "Additional_Payment", "End Balance"]]
    
    # Convert to a datetime object to make subsequent calcs easier
    schedule["Month"] = pd.to_datetime(schedule["Month"])
    
    #Create a summary statistics table
    payoff_date = schedule["Month"].iloc[-1]
    stats = pd.Series([payoff_date, schedule["Period"].count(), interest_rate,
                       years, principal, payment, addl_principal,
                       schedule["Interest"].sum()],
                       index=["Payoff Date", "Num Payments", "Interest Rate", "Years", "Principal",
                             "Payment", "Additional Payment", "Total Interest"])
    
    return schedule, stats
```

Example showing how to call the function


```python
df, stats = amortization_table(700000, .04, 30, addl_principal=200, start_date=date(2016, 1,1))
```


```python
stats
```




    Payoff Date           2042-12-01 00:00:00
    Num Payments                          324
    Interest Rate                        0.04
    Years                                  30
    Principal                          700000
    Payment                           3341.91
    Additional Payment                    200
    Total Interest                     444406
    dtype: object




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Period</th>
      <th>Month</th>
      <th>Begin Balance</th>
      <th>Payment</th>
      <th>Interest</th>
      <th>Principal</th>
      <th>Additional_Payment</th>
      <th>End Balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2016-01-01</td>
      <td>700000.00</td>
      <td>3341.91</td>
      <td>2333.33</td>
      <td>1008.58</td>
      <td>200.0</td>
      <td>698791.42</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2016-02-01</td>
      <td>698791.42</td>
      <td>3341.91</td>
      <td>2329.30</td>
      <td>1012.61</td>
      <td>200.0</td>
      <td>697578.81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2016-03-01</td>
      <td>697578.81</td>
      <td>3341.91</td>
      <td>2325.26</td>
      <td>1016.65</td>
      <td>200.0</td>
      <td>696362.16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2016-04-01</td>
      <td>696362.16</td>
      <td>3341.91</td>
      <td>2321.21</td>
      <td>1020.70</td>
      <td>200.0</td>
      <td>695141.46</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2016-05-01</td>
      <td>695141.46</td>
      <td>3341.91</td>
      <td>2317.14</td>
      <td>1024.77</td>
      <td>200.0</td>
      <td>693916.69</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Period</th>
      <th>Month</th>
      <th>Begin Balance</th>
      <th>Payment</th>
      <th>Interest</th>
      <th>Principal</th>
      <th>Additional_Payment</th>
      <th>End Balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>319</th>
      <td>320</td>
      <td>2042-08-01</td>
      <td>14413.65</td>
      <td>3341.91</td>
      <td>48.05</td>
      <td>3293.86</td>
      <td>200.0</td>
      <td>10919.79</td>
    </tr>
    <tr>
      <th>320</th>
      <td>321</td>
      <td>2042-09-01</td>
      <td>10919.79</td>
      <td>3341.91</td>
      <td>36.40</td>
      <td>3305.51</td>
      <td>200.0</td>
      <td>7414.28</td>
    </tr>
    <tr>
      <th>321</th>
      <td>322</td>
      <td>2042-10-01</td>
      <td>7414.28</td>
      <td>3341.91</td>
      <td>24.71</td>
      <td>3317.20</td>
      <td>200.0</td>
      <td>3897.08</td>
    </tr>
    <tr>
      <th>322</th>
      <td>323</td>
      <td>2042-11-01</td>
      <td>3897.08</td>
      <td>3341.91</td>
      <td>12.99</td>
      <td>3328.92</td>
      <td>200.0</td>
      <td>368.16</td>
    </tr>
    <tr>
      <th>323</th>
      <td>324</td>
      <td>2042-12-01</td>
      <td>368.16</td>
      <td>369.39</td>
      <td>1.23</td>
      <td>368.16</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



Make multiple calls to compare scenarios


```python
schedule1, stats1 = amortization_table(100000, .04, 30, addl_principal=50, start_date=date(2016,1,1))
schedule2, stats2 = amortization_table(100000, .05, 30, addl_principal=200, start_date=date(2016,1,1))
schedule3, stats3 = amortization_table(100000, .04, 15, addl_principal=0, start_date=date(2016,1,1))
```


```python
pd.DataFrame([stats1, stats2, stats3])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Payoff Date</th>
      <th>Num Payments</th>
      <th>Interest Rate</th>
      <th>Years</th>
      <th>Principal</th>
      <th>Payment</th>
      <th>Additional Payment</th>
      <th>Total Interest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2041-01-01</td>
      <td>301</td>
      <td>0.04</td>
      <td>30</td>
      <td>100000</td>
      <td>477.42</td>
      <td>50</td>
      <td>58441.08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2032-09-01</td>
      <td>201</td>
      <td>0.05</td>
      <td>30</td>
      <td>100000</td>
      <td>536.82</td>
      <td>200</td>
      <td>47708.38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2030-12-01</td>
      <td>180</td>
      <td>0.04</td>
      <td>15</td>
      <td>100000</td>
      <td>739.69</td>
      <td>0</td>
      <td>33143.79</td>
    </tr>
  </tbody>
</table>
</div>



Make some plots to show scenarios


```python
%matplotlib inline
plt.style.use('ggplot')
```


```python
fig, ax = plt.subplots(1, 1)
schedule1.plot(x='Month', y='End Balance', label="Scenario 1", ax=ax)
schedule2.plot(x='Month', y='End Balance', label="Scenario 2", ax=ax)
schedule3.plot(x='Month', y='End Balance', label="Scenario 3", ax=ax)
plt.title("Pay Off Timelines");
```


![png](Amortization%20Schedule_files/Amortization%20Schedule_15_0.png)



```python
def make_plot_data(schedule, stats):
    """Create a dataframe with annual interest totals, and a descriptive label"""
    y = schedule.set_index('Month')['Interest'].resample("A").sum().reset_index()
    y["Year"] = y["Month"].dt.year
    y.set_index('Year', inplace=True)
    y.drop('Month', 1, inplace=True)
    label="{} years at {}% with additional payment of ${}".format(stats['Years'], stats['Interest Rate']*100, stats['Additional Payment'])
    return y, label
    
y1, label1 = make_plot_data(schedule1, stats1)
y2, label2 = make_plot_data(schedule2, stats2)
y3, label3 = make_plot_data(schedule3, stats3)

y = pd.concat([y1, y2, y3], axis=1)
```


```python
figsize(7,5)
fig, ax = plt.subplots(1, 1)
y.plot(kind="bar", ax=ax)

plt.legend([label1, label2, label3], loc=1, prop={'size':10})
plt.title("Interest Payments");
```


![png](Amortization%20Schedule_files/Amortization%20Schedule_17_0.png)



```python
additional_payments = [0, 50, 200, 500]
fig, ax = plt.subplots(1, 1)

for pmt in additional_payments:
    result, _ = amortization_table(100000, .04, 30, addl_principal=pmt, start_date=date(2016,1,1))
    ax.plot(result['Month'], result['End Balance'], label='Addl Payment = ${}'.format(str(pmt)))
plt.title("Pay Off Timelines")
plt.ylabel("Balance")
ax.legend();
```


![png](Amortization%20Schedule_files/Amortization%20Schedule_18_0.png)

