

```python
import pandas as pd
## This is good to do for accounting values
pd.options.display.float_format = '{:,.2f}'.format
```


```python
gl = pd.read_excel(r'data/general-ledger-sample.xlsx')
```


```python
gl.head()
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
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th>Type</th>
      <th>Unnamed: 1</th>
      <th>Date</th>
      <th>Unnamed: 3</th>
      <th>Num</th>
      <th>Unnamed: 5</th>
      <th>Adj</th>
      <th>Unnamed: 7</th>
      <th>Name</th>
      <th>Unnamed: 9</th>
      <th>Memo</th>
      <th>Unnamed: 11</th>
      <th>Split</th>
      <th>Unnamed: 13</th>
      <th>Debit</th>
      <th>Unnamed: 15</th>
      <th>Credit</th>
      <th>Unnamed: 17</th>
      <th>Balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">NaN</th>
      <th>Company Checking Account</th>
      <th>NaN</th>
      <th>NaN</th>
      <th>NaN</th>
      <td>NaN</td>
      <td>nan</td>
      <td>NaT</td>
      <td>nan</td>
      <td>NaN</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>NaN</td>
      <td>nan</td>
      <td>NaN</td>
      <td>nan</td>
      <td>NaN</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>12,349.00</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">NaN</th>
      <th rowspan="4" valign="top">NaN</th>
      <th rowspan="4" valign="top">NaN</th>
      <th>NaN</th>
      <td>Check</td>
      <td>nan</td>
      <td>2018-01-01</td>
      <td>nan</td>
      <td>5001</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>Kuver Property</td>
      <td>nan</td>
      <td>Janaury Rent</td>
      <td>nan</td>
      <td>Rent</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>583.75</td>
      <td>nan</td>
      <td>11,765.25</td>
    </tr>
    <tr>
      <th>NaN</th>
      <td>Check</td>
      <td>nan</td>
      <td>2018-01-01</td>
      <td>nan</td>
      <td>5000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>Online Accounting</td>
      <td>nan</td>
      <td>Set up QuickBooks file</td>
      <td>nan</td>
      <td>Accounting Fees</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>225.00</td>
      <td>nan</td>
      <td>11,540.25</td>
    </tr>
    <tr>
      <th>NaN</th>
      <td>Deposit</td>
      <td>nan</td>
      <td>2018-01-01</td>
      <td>nan</td>
      <td>NaN</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>NaN</td>
      <td>nan</td>
      <td>Deposit</td>
      <td>nan</td>
      <td>-SPLIT-</td>
      <td>nan</td>
      <td>31,349.00</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>42,889.25</td>
    </tr>
    <tr>
      <th>NaN</th>
      <td>Check</td>
      <td>nan</td>
      <td>2018-01-05</td>
      <td>nan</td>
      <td>5002</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>Deborah Wood (Owner)</td>
      <td>nan</td>
      <td>Petty Cash</td>
      <td>nan</td>
      <td>Petty Cash Account</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>500.00</td>
      <td>nan</td>
      <td>42,389.25</td>
    </tr>
  </tbody>
</table>
</div>



A general ledger (GL) is a set of numbered accounts a business uses to keep track of its financial transactions and to prepare financial reports.


```python
## A parser to fix the above issue. 
def acct_append(row, new_accts):
    if pd.isnull(row[1]):
        new_accts.append(row[0])
    else:
        new_accts.append('{} | {}'.format(*row))


def fix_qb_gl(gl):
    gl = gl.dropna(axis=1, how='all')
    main_acct = list(gl.index.get_level_values(1))
    sub_acct = list(gl.index.get_level_values(2))
    acct = list(zip(main_acct, sub_acct))
    new_accts = []
    acct_append(acct[0], new_accts)
    
    for idx, (m, s) in enumerate(acct[1:]):
        if str(m).startswith('Total'):
            m = 'DELETE'       
        if str(s).startswith('Total'):
            s = 'DELETE'
        idx += 1
        acct[idx] = m, s

        if pd.isnull(m):  # Fill NA if main is NA
            acct[idx] = acct[idx - 1][0], acct[idx][1]
            
            if pd.isnull(s):  # If main is NA, then fill NA if sub is NA
                acct[idx] = acct[idx][0], acct[idx-1][1]

        acct_append(acct[idx], new_accts)  # Create the new acct
    gl = gl.reset_index(drop=True)
    gl['Acct'] = pd.Series(new_accts)
    gl[['Debit', 'Credit']] = gl[['Debit', 'Credit']].fillna(0)
    gl['Net'] = gl.apply(lambda x: (x['Debit'] - x['Credit']
                                    if 'DELETE' not in x['Acct']
                                    else 0), axis=1)
    gl = gl.fillna('NA')
    gl = gl.where(gl['Net'] != 0).dropna()
    columns = ['Acct', 'Type', 'Date', 'Num', 'Name', 'Memo',
               'Split', 'Debit', 'Credit', 'Net']
    gl = gl[columns]
    gl['Date'] = gl['Date'].apply(pd.datetime.date)
    return gl
```


```python
gl = fix_qb_gl(gl)
```

For the majority of the transactions, only one side Credit or Debit gets called on. The net effect gets reported on the right. Net is simply the cash inflow and outflow if your accounts are prepared on the cash basis. If your accounts are prepared on the accrual basis. 

Account on the left is the accounts that is credited or debited.

Name is the name of the person or company the transaction is associated with. Memo is a note to help improve the visibility of the transaction. Split is the contra account that is also influenced as part of the double ledger accounting system. The split lines are where the balances get set or recalibrated. 


```python
len(gl.Name.unique())
```




    111




```python
gl[gl.Acct == 'Inventory Asset'].head(5)
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
      <th>Acct</th>
      <th>Type</th>
      <th>Date</th>
      <th>Num</th>
      <th>Name</th>
      <th>Memo</th>
      <th>Split</th>
      <th>Debit</th>
      <th>Credit</th>
      <th>Net</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>624</th>
      <td>Inventory Asset</td>
      <td>Bill</td>
      <td>2018-01-02</td>
      <td>76850-2</td>
      <td>Peacock Bulb Manufacturing Company</td>
      <td>Specialty Bulbs, 100 watt (6 pack)</td>
      <td>Accounts Payable</td>
      <td>28.50</td>
      <td>0.00</td>
      <td>28.50</td>
    </tr>
    <tr>
      <th>625</th>
      <td>Inventory Asset</td>
      <td>Bill</td>
      <td>2018-01-02</td>
      <td>76850-2</td>
      <td>Peacock Bulb Manufacturing Company</td>
      <td>Cand. Light, 20 watts (8 pack)</td>
      <td>Accounts Payable</td>
      <td>48.00</td>
      <td>0.00</td>
      <td>48.00</td>
    </tr>
    <tr>
      <th>626</th>
      <td>Inventory Asset</td>
      <td>Bill</td>
      <td>2018-01-06</td>
      <td>87865</td>
      <td>Paulsen's Lighting</td>
      <td>Tiffany Collection,Golden Baroque, 1-light lan...</td>
      <td>Accounts Payable</td>
      <td>2,400.00</td>
      <td>0.00</td>
      <td>2,400.00</td>
    </tr>
    <tr>
      <th>627</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-06</td>
      <td>71047</td>
      <td>Baker's Professional Lighting:Store #25</td>
      <td>Pearl Nickle, 5 Light, Medium Base (100 watt max)</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>240.00</td>
      <td>-240.00</td>
    </tr>
    <tr>
      <th>628</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-06</td>
      <td>71047</td>
      <td>Baker's Professional Lighting:Store #25</td>
      <td>Black Finish, Solid Brass, Clear Glass, Hangin...</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>36.00</td>
      <td>-36.00</td>
    </tr>
  </tbody>
</table>
</div>



Now you can run some custom filter commands. For example here you filter the Inventory Asset and also the net amount. Only amounts above 6000 are printed.


```python
gl[gl.Acct == 'Inventory Asset'].where(abs(gl[gl.Acct == 'Inventory Asset'].Net) > 6000).dropna()
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
      <th>Acct</th>
      <th>Type</th>
      <th>Date</th>
      <th>Num</th>
      <th>Name</th>
      <th>Memo</th>
      <th>Split</th>
      <th>Debit</th>
      <th>Credit</th>
      <th>Net</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>794</th>
      <td>Inventory Asset</td>
      <td>Bill</td>
      <td>2018-03-09</td>
      <td>R909878</td>
      <td>Paulsen's Lighting</td>
      <td>Golden Umber (100 watt)  Six Light Chandelier</td>
      <td>Accounts Payable</td>
      <td>8,500.00</td>
      <td>0.00</td>
      <td>8,500.00</td>
    </tr>
    <tr>
      <th>795</th>
      <td>Inventory Asset</td>
      <td>Bill</td>
      <td>2018-03-09</td>
      <td>R909878</td>
      <td>Paulsen's Lighting</td>
      <td>Burnished Brass (60 watt)  w/Golden Umber Accents</td>
      <td>Accounts Payable</td>
      <td>7,500.00</td>
      <td>0.00</td>
      <td>7,500.00</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>Inventory Asset</td>
      <td>Bill</td>
      <td>2018-09-06</td>
      <td>6785</td>
      <td>Hall Lighting &amp; Accessories</td>
      <td>Sunset, Clear Glass, 1-100 watt max.</td>
      <td>Accounts Payable</td>
      <td>7,500.00</td>
      <td>0.00</td>
      <td>7,500.00</td>
    </tr>
  </tbody>
</table>
</div>



Here is an example of filtering between certain dates.


```python
start_date = pd.datetime(2018,1,27).date()
end_date = pd.datetime(2018,1,28).date()
gl.where((start_date <= gl.Date) & (gl.Date <= end_date)).dropna()
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
      <th>Acct</th>
      <th>Type</th>
      <th>Date</th>
      <th>Num</th>
      <th>Name</th>
      <th>Memo</th>
      <th>Split</th>
      <th>Debit</th>
      <th>Credit</th>
      <th>Net</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>444</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71053</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>854.00</td>
      <td>0.00</td>
      <td>854.00</td>
    </tr>
    <tr>
      <th>445</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>2,187.45</td>
      <td>0.00</td>
      <td>2,187.45</td>
    </tr>
    <tr>
      <th>446</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-01-28</td>
      <td>254</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>71053</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>854.00</td>
      <td>-854.00</td>
    </tr>
    <tr>
      <th>670</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71053</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>Tapestry, (2-60 watt Med), Etched Cracked Glas...</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>30.00</td>
      <td>-30.00</td>
    </tr>
    <tr>
      <th>671</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71053</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>Tapestry (8/60 watt)  Faux Alabaster Glass</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>75.00</td>
      <td>-75.00</td>
    </tr>
    <tr>
      <th>672</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71053</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>Vianne Lanterns, Satin Antiqued Solid Brass, 3...</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>193.79</td>
      <td>-193.79</td>
    </tr>
    <tr>
      <th>673</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71053</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>White, 4.5x7.5, 1-100 watt Medium Base, Satin ...</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>30.00</td>
      <td>-30.00</td>
    </tr>
    <tr>
      <th>674</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Die Cast Lanterns, Black, 1-100 watt, Medium B...</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>228.00</td>
      <td>-228.00</td>
    </tr>
    <tr>
      <th>675</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Tapestry, (2-60 watt Med), Etched Cracked Glas...</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>90.00</td>
      <td>-90.00</td>
    </tr>
    <tr>
      <th>676</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Pearl Nickle, 5 Light, Medium Base (100 watt max)</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>160.00</td>
      <td>-160.00</td>
    </tr>
    <tr>
      <th>677</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Sunset, Clear Glass, 1-100 watt max.</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>60.00</td>
      <td>-60.00</td>
    </tr>
    <tr>
      <th>678</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Black Finish, Solid Brass, Clear Glass, Hangin...</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>54.00</td>
      <td>-54.00</td>
    </tr>
    <tr>
      <th>679</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Burnished Brass (60 watt)  w/Golden Umber Accents</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>75.00</td>
      <td>-75.00</td>
    </tr>
    <tr>
      <th>680</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Chestnut (3/60 watt)  Marble Glass</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>95.00</td>
      <td>-95.00</td>
    </tr>
    <tr>
      <th>681</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Pendant Bar, Textured White, 50lbs max</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>10.00</td>
      <td>-10.00</td>
    </tr>
    <tr>
      <th>682</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Athena Collection, (100 watt max) Copper Verde...</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>32.14</td>
      <td>-32.14</td>
    </tr>
    <tr>
      <th>683</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Cand. Light, 20 watts (8 pack)</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>24.00</td>
      <td>-24.00</td>
    </tr>
    <tr>
      <th>684</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Fluorescent Lamp, T-12, Medium Bipin  (30pack)</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>6.34</td>
      <td>-6.34</td>
    </tr>
    <tr>
      <th>685</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Fluorescent Lamp, T-5, Bipin</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>6.00</td>
      <td>-6.00</td>
    </tr>
    <tr>
      <th>686</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Halogen Lamp, Volts:3.5  Tubular</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>2.50</td>
      <td>-2.50</td>
    </tr>
    <tr>
      <th>687</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Specialty Bulbs, 100 watt (6 pack)</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>9.50</td>
      <td>-9.50</td>
    </tr>
    <tr>
      <th>688</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Specialty, Stage &amp; Studio Bulbs  60 watt</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>20.00</td>
      <td>-20.00</td>
    </tr>
    <tr>
      <th>689</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Flat Glass, (25 watt max) Polished Brass, 4 light</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>40.00</td>
      <td>-40.00</td>
    </tr>
    <tr>
      <th>690</th>
      <td>Inventory Asset</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Beveled Mirror, Etched Ribbed Glass (75 watt max)</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>44.00</td>
      <td>-44.00</td>
    </tr>
    <tr>
      <th>1581</th>
      <td>Undeposited Funds</td>
      <td>Payment</td>
      <td>2018-01-28</td>
      <td>254</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>71053</td>
      <td>Accounts Receivable</td>
      <td>854.00</td>
      <td>0.00</td>
      <td>854.00</td>
    </tr>
    <tr>
      <th>1871</th>
      <td>QuickBooks Credit Card</td>
      <td>Credit Card Charge</td>
      <td>2018-01-28</td>
      <td>1256831</td>
      <td>Business Supply Center</td>
      <td>ink cartridges</td>
      <td>Supplies</td>
      <td>0.00</td>
      <td>18.15</td>
      <td>-18.15</td>
    </tr>
    <tr>
      <th>2913</th>
      <td>Revenue | Revenue</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71053</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>Tapestry, (2-60 watt Med), Etched Cracked Glas...</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>90.00</td>
      <td>-90.00</td>
    </tr>
    <tr>
      <th>2914</th>
      <td>Revenue | Revenue</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71053</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>Tapestry (8/60 watt)  Faux Alabaster Glass</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>300.00</td>
      <td>-300.00</td>
    </tr>
    <tr>
      <th>2915</th>
      <td>Revenue | Revenue</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71053</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>Vianne Lanterns, Satin Antiqued Solid Brass, 3...</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>400.00</td>
      <td>-400.00</td>
    </tr>
    <tr>
      <th>2916</th>
      <td>Revenue | Revenue</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71053</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>White, 4.5x7.5, 1-100 watt Medium Base, Satin ...</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>64.00</td>
      <td>-64.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2926</th>
      <td>Revenue | Revenue</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Cand. Light, 20 watts (8 pack)</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>50.40</td>
      <td>-50.40</td>
    </tr>
    <tr>
      <th>2927</th>
      <td>Revenue | Revenue</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Fluorescent Lamp, T-12, Medium Bipin  (30pack)</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>13.50</td>
      <td>-13.50</td>
    </tr>
    <tr>
      <th>2928</th>
      <td>Revenue | Revenue</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Fluorescent Lamp, T-5, Bipin</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>18.00</td>
      <td>-18.00</td>
    </tr>
    <tr>
      <th>2929</th>
      <td>Revenue | Revenue</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Halogen Lamp, Volts:3.5  Tubular</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>4.95</td>
      <td>-4.95</td>
    </tr>
    <tr>
      <th>2930</th>
      <td>Revenue | Revenue</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Specialty Bulbs, 100 watt (6 pack)</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>19.80</td>
      <td>-19.80</td>
    </tr>
    <tr>
      <th>2931</th>
      <td>Revenue | Revenue</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Specialty, Stage &amp; Studio Bulbs  60 watt</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>46.80</td>
      <td>-46.80</td>
    </tr>
    <tr>
      <th>2932</th>
      <td>Revenue | Revenue</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Flat Glass, (25 watt max) Polished Brass, 4 light</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>90.00</td>
      <td>-90.00</td>
    </tr>
    <tr>
      <th>2933</th>
      <td>Revenue | Revenue</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Beveled Mirror, Etched Ribbed Glass (75 watt max)</td>
      <td>Accounts Receivable</td>
      <td>0.00</td>
      <td>86.40</td>
      <td>-86.40</td>
    </tr>
    <tr>
      <th>3841</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71053</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>Tapestry, (2-60 watt Med), Etched Cracked Glas...</td>
      <td>Accounts Receivable</td>
      <td>30.00</td>
      <td>0.00</td>
      <td>30.00</td>
    </tr>
    <tr>
      <th>3842</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71053</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>Tapestry (8/60 watt)  Faux Alabaster Glass</td>
      <td>Accounts Receivable</td>
      <td>75.00</td>
      <td>0.00</td>
      <td>75.00</td>
    </tr>
    <tr>
      <th>3843</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71053</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>Vianne Lanterns, Satin Antiqued Solid Brass, 3...</td>
      <td>Accounts Receivable</td>
      <td>193.79</td>
      <td>0.00</td>
      <td>193.79</td>
    </tr>
    <tr>
      <th>3844</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71053</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>White, 4.5x7.5, 1-100 watt Medium Base, Satin ...</td>
      <td>Accounts Receivable</td>
      <td>30.00</td>
      <td>0.00</td>
      <td>30.00</td>
    </tr>
    <tr>
      <th>3845</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Die Cast Lanterns, Black, 1-100 watt, Medium B...</td>
      <td>Accounts Receivable</td>
      <td>228.00</td>
      <td>0.00</td>
      <td>228.00</td>
    </tr>
    <tr>
      <th>3846</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Tapestry, (2-60 watt Med), Etched Cracked Glas...</td>
      <td>Accounts Receivable</td>
      <td>90.00</td>
      <td>0.00</td>
      <td>90.00</td>
    </tr>
    <tr>
      <th>3847</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Pearl Nickle, 5 Light, Medium Base (100 watt max)</td>
      <td>Accounts Receivable</td>
      <td>160.00</td>
      <td>0.00</td>
      <td>160.00</td>
    </tr>
    <tr>
      <th>3848</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Sunset, Clear Glass, 1-100 watt max.</td>
      <td>Accounts Receivable</td>
      <td>60.00</td>
      <td>0.00</td>
      <td>60.00</td>
    </tr>
    <tr>
      <th>3849</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Black Finish, Solid Brass, Clear Glass, Hangin...</td>
      <td>Accounts Receivable</td>
      <td>54.00</td>
      <td>0.00</td>
      <td>54.00</td>
    </tr>
    <tr>
      <th>3850</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Burnished Brass (60 watt)  w/Golden Umber Accents</td>
      <td>Accounts Receivable</td>
      <td>75.00</td>
      <td>0.00</td>
      <td>75.00</td>
    </tr>
    <tr>
      <th>3851</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Chestnut (3/60 watt)  Marble Glass</td>
      <td>Accounts Receivable</td>
      <td>95.00</td>
      <td>0.00</td>
      <td>95.00</td>
    </tr>
    <tr>
      <th>3852</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Pendant Bar, Textured White, 50lbs max</td>
      <td>Accounts Receivable</td>
      <td>10.00</td>
      <td>0.00</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>3853</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Athena Collection, (100 watt max) Copper Verde...</td>
      <td>Accounts Receivable</td>
      <td>32.14</td>
      <td>0.00</td>
      <td>32.14</td>
    </tr>
    <tr>
      <th>3854</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Cand. Light, 20 watts (8 pack)</td>
      <td>Accounts Receivable</td>
      <td>24.00</td>
      <td>0.00</td>
      <td>24.00</td>
    </tr>
    <tr>
      <th>3855</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Fluorescent Lamp, T-12, Medium Bipin  (30pack)</td>
      <td>Accounts Receivable</td>
      <td>6.34</td>
      <td>0.00</td>
      <td>6.34</td>
    </tr>
    <tr>
      <th>3856</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Fluorescent Lamp, T-5, Bipin</td>
      <td>Accounts Receivable</td>
      <td>6.00</td>
      <td>0.00</td>
      <td>6.00</td>
    </tr>
    <tr>
      <th>3857</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Halogen Lamp, Volts:3.5  Tubular</td>
      <td>Accounts Receivable</td>
      <td>2.50</td>
      <td>0.00</td>
      <td>2.50</td>
    </tr>
    <tr>
      <th>3858</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Specialty Bulbs, 100 watt (6 pack)</td>
      <td>Accounts Receivable</td>
      <td>9.50</td>
      <td>0.00</td>
      <td>9.50</td>
    </tr>
    <tr>
      <th>3859</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Specialty, Stage &amp; Studio Bulbs  60 watt</td>
      <td>Accounts Receivable</td>
      <td>20.00</td>
      <td>0.00</td>
      <td>20.00</td>
    </tr>
    <tr>
      <th>3860</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Flat Glass, (25 watt max) Polished Brass, 4 light</td>
      <td>Accounts Receivable</td>
      <td>40.00</td>
      <td>0.00</td>
      <td>40.00</td>
    </tr>
    <tr>
      <th>3861</th>
      <td>Purchases  (Cost of Goods)</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>Beveled Mirror, Etched Ribbed Glass (75 watt max)</td>
      <td>Accounts Receivable</td>
      <td>44.00</td>
      <td>0.00</td>
      <td>44.00</td>
    </tr>
    <tr>
      <th>4991</th>
      <td>Supplies</td>
      <td>Credit Card Charge</td>
      <td>2018-01-28</td>
      <td>1256831</td>
      <td>Business Supply Center</td>
      <td>ink cartridges</td>
      <td>QuickBooks Credit Card</td>
      <td>18.15</td>
      <td>0.00</td>
      <td>18.15</td>
    </tr>
  </tbody>
</table>
<p>69 rows × 10 columns</p>
</div>



This is simply a pivot table that sums up each unique accounts. 


```python
pivot = gl.pivot_table(values=['Debit', 'Credit', 'Net'], index='Acct', aggfunc='sum', margins=True)
accts = list(gl.Acct.unique())
accts.append('All')
pivot.loc[accts]
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
      <th>Credit</th>
      <th>Debit</th>
      <th>Net</th>
    </tr>
    <tr>
      <th>Acct</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Company Checking Account</th>
      <td>403,171.13</td>
      <td>480,976.45</td>
      <td>77,805.32</td>
    </tr>
    <tr>
      <th>Petty Cash Account</th>
      <td>0.00</td>
      <td>500.00</td>
      <td>500.00</td>
    </tr>
    <tr>
      <th>Accounts Receivable</th>
      <td>375,976.45</td>
      <td>408,309.53</td>
      <td>32,333.08</td>
    </tr>
    <tr>
      <th>Inventory Asset</th>
      <td>173,360.75</td>
      <td>131,301.50</td>
      <td>-42,059.25</td>
    </tr>
    <tr>
      <th>Prepaids | Prepaid Insurance</th>
      <td>6,875.00</td>
      <td>6,875.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Undeposited Funds</th>
      <td>375,976.45</td>
      <td>375,976.45</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Computer &amp; Office Equipment</th>
      <td>0.00</td>
      <td>2,500.00</td>
      <td>2,500.00</td>
    </tr>
    <tr>
      <th>Accumulated Depreciation</th>
      <td>923.04</td>
      <td>0.00</td>
      <td>-923.04</td>
    </tr>
    <tr>
      <th>Accounts Payable</th>
      <td>147,728.80</td>
      <td>131,031.50</td>
      <td>-16,697.30</td>
    </tr>
    <tr>
      <th>QuickBooks Credit Card</th>
      <td>3,453.60</td>
      <td>2,204.48</td>
      <td>-1,249.12</td>
    </tr>
    <tr>
      <th>Customer Deposits</th>
      <td>0.00</td>
      <td>3,500.00</td>
      <td>3,500.00</td>
    </tr>
    <tr>
      <th>Line of Credit</th>
      <td>106,208.85</td>
      <td>25,000.00</td>
      <td>-81,208.85</td>
    </tr>
    <tr>
      <th>Payroll Liabilities | Payroll Taxes Payable</th>
      <td>14,251.22</td>
      <td>11,377.50</td>
      <td>-2,873.72</td>
    </tr>
    <tr>
      <th>Payroll Liabilities | SEC125 Payable</th>
      <td>1,400.00</td>
      <td>1,350.00</td>
      <td>-50.00</td>
    </tr>
    <tr>
      <th>Deborah Wood Equity | Deborah Wood's Time to Jobs</th>
      <td>114,450.00</td>
      <td>114,450.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Deborah Wood Equity | Deborah Wood Draws</th>
      <td>0.00</td>
      <td>135,000.00</td>
      <td>135,000.00</td>
    </tr>
    <tr>
      <th>Opening Bal Equity</th>
      <td>375.00</td>
      <td>0.00</td>
      <td>-375.00</td>
    </tr>
    <tr>
      <th>Revenue | Revenue</th>
      <td>411,809.53</td>
      <td>0.00</td>
      <td>-411,809.53</td>
    </tr>
    <tr>
      <th>Direct Labor | Wages - Sales-Inside</th>
      <td>0.00</td>
      <td>2,500.00</td>
      <td>2,500.00</td>
    </tr>
    <tr>
      <th>Direct Labor | Wages - Warehouse</th>
      <td>0.00</td>
      <td>19,705.00</td>
      <td>19,705.00</td>
    </tr>
    <tr>
      <th>Freight Costs</th>
      <td>0.00</td>
      <td>1,810.00</td>
      <td>1,810.00</td>
    </tr>
    <tr>
      <th>Packaging Materials</th>
      <td>0.00</td>
      <td>1,752.50</td>
      <td>1,752.50</td>
    </tr>
    <tr>
      <th>Purchases  (Cost of Goods)</th>
      <td>69.00</td>
      <td>180,030.75</td>
      <td>179,961.75</td>
    </tr>
    <tr>
      <th>Sales Commission (outside reps)</th>
      <td>0.00</td>
      <td>3,990.80</td>
      <td>3,990.80</td>
    </tr>
    <tr>
      <th>Advertising Expense</th>
      <td>0.00</td>
      <td>2,000.00</td>
      <td>2,000.00</td>
    </tr>
    <tr>
      <th>Business License &amp; Fees</th>
      <td>0.00</td>
      <td>710.23</td>
      <td>710.23</td>
    </tr>
    <tr>
      <th>Car/Truck Expense | Car Lease</th>
      <td>0.00</td>
      <td>6,756.00</td>
      <td>6,756.00</td>
    </tr>
    <tr>
      <th>Car/Truck Expense | Gas</th>
      <td>0.00</td>
      <td>907.64</td>
      <td>907.64</td>
    </tr>
    <tr>
      <th>Car/Truck Expense | Insurance-Auto</th>
      <td>0.00</td>
      <td>1,440.00</td>
      <td>1,440.00</td>
    </tr>
    <tr>
      <th>Car/Truck Expense | Registration &amp; License</th>
      <td>0.00</td>
      <td>546.00</td>
      <td>546.00</td>
    </tr>
    <tr>
      <th>Car/Truck Expense | Repairs &amp; Maintenance</th>
      <td>0.00</td>
      <td>1,700.23</td>
      <td>1,700.23</td>
    </tr>
    <tr>
      <th>Conferences and Seminars</th>
      <td>0.00</td>
      <td>4,700.00</td>
      <td>4,700.00</td>
    </tr>
    <tr>
      <th>Contributions</th>
      <td>0.00</td>
      <td>2,500.00</td>
      <td>2,500.00</td>
    </tr>
    <tr>
      <th>Depreciation Expense</th>
      <td>0.00</td>
      <td>923.04</td>
      <td>923.04</td>
    </tr>
    <tr>
      <th>Dues and Subscriptions</th>
      <td>0.00</td>
      <td>1,900.00</td>
      <td>1,900.00</td>
    </tr>
    <tr>
      <th>Insurance | General Liability Insurance</th>
      <td>0.00</td>
      <td>2,760.00</td>
      <td>2,760.00</td>
    </tr>
    <tr>
      <th>Insurance | Owner's Health Insurance</th>
      <td>0.00</td>
      <td>4,272.00</td>
      <td>4,272.00</td>
    </tr>
    <tr>
      <th>Insurance | Professional Liability Insuranc</th>
      <td>0.00</td>
      <td>6,875.00</td>
      <td>6,875.00</td>
    </tr>
    <tr>
      <th>Insurance | Worker's Compensation</th>
      <td>0.00</td>
      <td>2,782.08</td>
      <td>2,782.08</td>
    </tr>
    <tr>
      <th>Maintenance/Janitorial</th>
      <td>0.00</td>
      <td>2,841.95</td>
      <td>2,841.95</td>
    </tr>
    <tr>
      <th>Marketing Expense</th>
      <td>0.00</td>
      <td>4,982.00</td>
      <td>4,982.00</td>
    </tr>
    <tr>
      <th>Meals and Entertainment</th>
      <td>0.00</td>
      <td>1,376.35</td>
      <td>1,376.35</td>
    </tr>
    <tr>
      <th>Office Equipment</th>
      <td>0.00</td>
      <td>1,100.00</td>
      <td>1,100.00</td>
    </tr>
    <tr>
      <th>Postage and Delivery</th>
      <td>0.00</td>
      <td>1,098.00</td>
      <td>1,098.00</td>
    </tr>
    <tr>
      <th>Professional Fees | Accounting Fees</th>
      <td>0.00</td>
      <td>2,544.00</td>
      <td>2,544.00</td>
    </tr>
    <tr>
      <th>Professional Fees | Legal Fees</th>
      <td>0.00</td>
      <td>600.00</td>
      <td>600.00</td>
    </tr>
    <tr>
      <th>Professional Fees | Payroll Service Fees</th>
      <td>0.00</td>
      <td>1,529.24</td>
      <td>1,529.24</td>
    </tr>
    <tr>
      <th>Promotional Expense</th>
      <td>0.00</td>
      <td>2,021.00</td>
      <td>2,021.00</td>
    </tr>
    <tr>
      <th>Rent</th>
      <td>0.00</td>
      <td>7,005.00</td>
      <td>7,005.00</td>
    </tr>
    <tr>
      <th>Repairs | Computer Repairs</th>
      <td>0.00</td>
      <td>390.00</td>
      <td>390.00</td>
    </tr>
    <tr>
      <th>Supplies</th>
      <td>0.00</td>
      <td>6,199.36</td>
      <td>6,199.36</td>
    </tr>
    <tr>
      <th>Telephone</th>
      <td>0.00</td>
      <td>4,003.44</td>
      <td>4,003.44</td>
    </tr>
    <tr>
      <th>Travel</th>
      <td>0.00</td>
      <td>3,452.23</td>
      <td>3,452.23</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>0.00</td>
      <td>501.59</td>
      <td>501.59</td>
    </tr>
    <tr>
      <th>Wages | Employee Benefits</th>
      <td>0.00</td>
      <td>2,253.96</td>
      <td>2,253.96</td>
    </tr>
    <tr>
      <th>Wages | Payroll Tax Expenses</th>
      <td>0.00</td>
      <td>4,608.57</td>
      <td>4,608.57</td>
    </tr>
    <tr>
      <th>Wages | Wages - Office Staff</th>
      <td>0.00</td>
      <td>6,312.00</td>
      <td>6,312.00</td>
    </tr>
    <tr>
      <th>Other Expense | Interest Expense</th>
      <td>0.00</td>
      <td>2,296.45</td>
      <td>2,296.45</td>
    </tr>
    <tr>
      <th>All</th>
      <td>2,136,028.82</td>
      <td>2,136,028.82</td>
      <td>-0.00</td>
    </tr>
  </tbody>
</table>
</div>



Now of course there is countless other things you can do, the possibilities are endless, I will do three of them, and then give a few more ideas for those who which to further extend this notebook


```python
#aggregation by type. This approach below 
```


```python
gl.groupby("Type").sum()
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
      <th>Debit</th>
      <th>Credit</th>
      <th>Net</th>
    </tr>
    <tr>
      <th>Type</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bill</th>
      <td>147,802.80</td>
      <td>147,802.80</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Bill Pmt -Check</th>
      <td>131,031.50</td>
      <td>131,031.50</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Check</th>
      <td>216,937.78</td>
      <td>216,937.78</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Credit Card Charge</th>
      <td>3,453.60</td>
      <td>3,453.60</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Deposit</th>
      <td>375,976.45</td>
      <td>375,976.45</td>
      <td>-0.00</td>
    </tr>
    <tr>
      <th>General Journal</th>
      <td>9,006.89</td>
      <td>9,006.89</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Inventory Adjust</th>
      <td>375.00</td>
      <td>375.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Invoice</th>
      <td>585,165.28</td>
      <td>585,165.28</td>
      <td>-0.00</td>
    </tr>
    <tr>
      <th>Liability Check</th>
      <td>12,727.50</td>
      <td>12,727.50</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Paycheck</th>
      <td>147,575.57</td>
      <td>147,575.57</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Payment</th>
      <td>375,976.45</td>
      <td>375,976.45</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Transfer</th>
      <td>130,000.00</td>
      <td>130,000.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



Below is an interesting insight as it seems that their has been some contra accounts that have been mistakly categorised.


```python
len(gl.Acct.unique())
```




    58




```python
len(gl.Split.unique())
```




    49




```python
ar = gl[gl["Acct"]=="Accounts Receivable"]; ar
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
      <th>Acct</th>
      <th>Type</th>
      <th>Date</th>
      <th>Num</th>
      <th>Name</th>
      <th>Memo</th>
      <th>Split</th>
      <th>Debit</th>
      <th>Credit</th>
      <th>Net</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>438</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-01-06</td>
      <td>71047</td>
      <td>Baker's Professional Lighting:Store #25</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>2,381.00</td>
      <td>0.00</td>
      <td>2,381.00</td>
    </tr>
    <tr>
      <th>439</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-01-14</td>
      <td>71050</td>
      <td>Godwin Lighting Depot:Store #202</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>7,786.40</td>
      <td>0.00</td>
      <td>7,786.40</td>
    </tr>
    <tr>
      <th>440</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-01-17</td>
      <td>71055</td>
      <td>Miscellaneous - Retail:Ms. Jann Minor</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,084.00</td>
      <td>0.00</td>
      <td>1,084.00</td>
    </tr>
    <tr>
      <th>441</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-01-17</td>
      <td>555</td>
      <td>Miscellaneous - Retail:Ms. Jann Minor</td>
      <td>71055</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>1,084.00</td>
      <td>-1,084.00</td>
    </tr>
    <tr>
      <th>442</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-01-18</td>
      <td>71092</td>
      <td>Miscellaneous - Retail:Brian Stern</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,126.00</td>
      <td>0.00</td>
      <td>1,126.00</td>
    </tr>
    <tr>
      <th>443</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-01-18</td>
      <td>11124</td>
      <td>Miscellaneous - Retail:Brian Stern</td>
      <td>71092</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>1,126.00</td>
      <td>-1,126.00</td>
    </tr>
    <tr>
      <th>444</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71053</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>854.00</td>
      <td>0.00</td>
      <td>854.00</td>
    </tr>
    <tr>
      <th>445</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-01-28</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>2,187.45</td>
      <td>0.00</td>
      <td>2,187.45</td>
    </tr>
    <tr>
      <th>446</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-01-28</td>
      <td>254</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>71053</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>854.00</td>
      <td>-854.00</td>
    </tr>
    <tr>
      <th>447</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-01-29</td>
      <td>71124</td>
      <td>Kern Lighting Warehouse:Store #34</td>
      <td>NA</td>
      <td>Revenue</td>
      <td>8,400.00</td>
      <td>0.00</td>
      <td>8,400.00</td>
    </tr>
    <tr>
      <th>448</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-01-31</td>
      <td>71059</td>
      <td>Godwin Lighting Depot:Store #303</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,687.95</td>
      <td>0.00</td>
      <td>1,687.95</td>
    </tr>
    <tr>
      <th>449</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-02-01</td>
      <td>71121</td>
      <td>Kern Lighting Warehouse:Store #34</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>6,745.00</td>
      <td>0.00</td>
      <td>6,745.00</td>
    </tr>
    <tr>
      <th>450</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-02-09</td>
      <td>130</td>
      <td>Baker's Professional Lighting:Store #25</td>
      <td>71047</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>2,381.00</td>
      <td>-2,381.00</td>
    </tr>
    <tr>
      <th>451</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-02-10</td>
      <td>71051</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>4,364.00</td>
      <td>0.00</td>
      <td>4,364.00</td>
    </tr>
    <tr>
      <th>452</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-02-11</td>
      <td>71112</td>
      <td>Thompson Lighting Stores:Store #15</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>3,025.00</td>
      <td>0.00</td>
      <td>3,025.00</td>
    </tr>
    <tr>
      <th>453</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-02-12</td>
      <td>71088</td>
      <td>Cole Home Builders:Phase 1 - Lot 2</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>2,187.45</td>
      <td>0.00</td>
      <td>2,187.45</td>
    </tr>
    <tr>
      <th>454</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-02-14</td>
      <td>71110</td>
      <td>Baker's Professional Lighting:Store #05</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>2,391.00</td>
      <td>0.00</td>
      <td>2,391.00</td>
    </tr>
    <tr>
      <th>455</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-02-15</td>
      <td>1103</td>
      <td>Stern Commercial Contractor's:Walker Properties</td>
      <td>71106</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>21,330.00</td>
      <td>-21,330.00</td>
    </tr>
    <tr>
      <th>456</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-02-18</td>
      <td>71093</td>
      <td>Miscellaneous - Retail:Ruth Kuver</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,366.00</td>
      <td>0.00</td>
      <td>1,366.00</td>
    </tr>
    <tr>
      <th>457</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-02-18</td>
      <td>5467</td>
      <td>Miscellaneous - Retail:Ruth Kuver</td>
      <td>71093</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>1,366.00</td>
      <td>-1,366.00</td>
    </tr>
    <tr>
      <th>458</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-02-20</td>
      <td>71054</td>
      <td>Miscellaneous - Retail:Mrs. Chris Holly</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,300.00</td>
      <td>0.00</td>
      <td>1,300.00</td>
    </tr>
    <tr>
      <th>459</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-02-20</td>
      <td>305</td>
      <td>Miscellaneous - Retail:Mrs. Chris Holly</td>
      <td>71054</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>1,300.00</td>
      <td>-1,300.00</td>
    </tr>
    <tr>
      <th>460</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-02-23</td>
      <td>71094</td>
      <td>Miscellaneous - Retail:David Lo</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,932.00</td>
      <td>0.00</td>
      <td>1,932.00</td>
    </tr>
    <tr>
      <th>461</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-02-23</td>
      <td>12357</td>
      <td>Miscellaneous - Retail:David Lo</td>
      <td>71094</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>1,932.00</td>
      <td>-1,932.00</td>
    </tr>
    <tr>
      <th>462</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-02-27</td>
      <td>71052</td>
      <td>Baker's Professional Lighting:Store #15</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,885.00</td>
      <td>0.00</td>
      <td>1,885.00</td>
    </tr>
    <tr>
      <th>463</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-02-28</td>
      <td>71122</td>
      <td>Lavery Lighting &amp; Design:Store #JL-08</td>
      <td>NA</td>
      <td>Revenue</td>
      <td>600.00</td>
      <td>0.00</td>
      <td>600.00</td>
    </tr>
    <tr>
      <th>464</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-02-28</td>
      <td>57702</td>
      <td>Godwin Lighting Depot:Store #202</td>
      <td>71050</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>7,786.40</td>
      <td>-7,786.40</td>
    </tr>
    <tr>
      <th>465</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-02-28</td>
      <td>57713</td>
      <td>Godwin Lighting Depot:Store #303</td>
      <td>71059</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>1,687.95</td>
      <td>-1,687.95</td>
    </tr>
    <tr>
      <th>466</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-03-02</td>
      <td>5749</td>
      <td>Kern Lighting Warehouse:Store #34</td>
      <td>71124</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>8,400.00</td>
      <td>-8,400.00</td>
    </tr>
    <tr>
      <th>467</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-03-02</td>
      <td>5750</td>
      <td>Kern Lighting Warehouse:Store #13</td>
      <td>71123</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>110.00</td>
      <td>-110.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>590</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-11-05</td>
      <td>71117</td>
      <td>Lavery Lighting &amp; Design:Store #JL-01</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>5,279.00</td>
      <td>0.00</td>
      <td>5,279.00</td>
    </tr>
    <tr>
      <th>591</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-11-05</td>
      <td>101</td>
      <td>Miscellaneous - Retail:Valesha Jones</td>
      <td>71103</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>2,395.00</td>
      <td>-2,395.00</td>
    </tr>
    <tr>
      <th>592</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-11-16</td>
      <td>71118</td>
      <td>Lavery Lighting &amp; Design:Store #JL-04</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>11,715.00</td>
      <td>0.00</td>
      <td>11,715.00</td>
    </tr>
    <tr>
      <th>593</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-11-16</td>
      <td>11270</td>
      <td>Cole Home Builders:Phase 1 - Lot 5</td>
      <td>71073</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>2,138.40</td>
      <td>-2,138.40</td>
    </tr>
    <tr>
      <th>594</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-11-16</td>
      <td>11257</td>
      <td>Cole Home Builders:Phase 1 - Lot 5</td>
      <td>71074</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>2,138.40</td>
      <td>-2,138.40</td>
    </tr>
    <tr>
      <th>595</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-11-16</td>
      <td>11250</td>
      <td>Cole Home Builders:Phase 1 - Lot 5</td>
      <td>71075</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>2,138.40</td>
      <td>-2,138.40</td>
    </tr>
    <tr>
      <th>596</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-11-16</td>
      <td>11280</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>71091</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>2,138.40</td>
      <td>-2,138.40</td>
    </tr>
    <tr>
      <th>597</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-11-16</td>
      <td>5930</td>
      <td>Kern Lighting Warehouse:Store #13</td>
      <td>71131</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>5,569.00</td>
      <td>-5,569.00</td>
    </tr>
    <tr>
      <th>598</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-11-20</td>
      <td>71104</td>
      <td>Miscellaneous - Retail:Melanie Hall</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,600.00</td>
      <td>0.00</td>
      <td>1,600.00</td>
    </tr>
    <tr>
      <th>599</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-11-20</td>
      <td>55449</td>
      <td>Miscellaneous - Retail:Melanie Hall</td>
      <td>71104</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>1,600.00</td>
      <td>-1,600.00</td>
    </tr>
    <tr>
      <th>600</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-11-22</td>
      <td>71133</td>
      <td>Kern Lighting Warehouse:Store #01</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>9,839.00</td>
      <td>0.00</td>
      <td>9,839.00</td>
    </tr>
    <tr>
      <th>601</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-11-29</td>
      <td>71123</td>
      <td>Kern Lighting Warehouse:Store #13</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>110.00</td>
      <td>0.00</td>
      <td>110.00</td>
    </tr>
    <tr>
      <th>602</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-11-29</td>
      <td>1052</td>
      <td>Stern Commercial Contractor's:Tittle Properties</td>
      <td>71107</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>17,433.00</td>
      <td>-17,433.00</td>
    </tr>
    <tr>
      <th>603</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-11-30</td>
      <td>5791</td>
      <td>Kern Lighting Warehouse:Store #13</td>
      <td>71120</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>8,190.00</td>
      <td>-8,190.00</td>
    </tr>
    <tr>
      <th>604</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-11-30</td>
      <td>1098</td>
      <td>Stern Commercial Contractor's:Wilson Suites</td>
      <td>71108</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>14,355.00</td>
      <td>-14,355.00</td>
    </tr>
    <tr>
      <th>605</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-12-01</td>
      <td>11311</td>
      <td>Lavery Lighting &amp; Design:Store #JL-04</td>
      <td>71118</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>11,715.00</td>
      <td>-11,715.00</td>
    </tr>
    <tr>
      <th>606</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-12-03</td>
      <td>71139</td>
      <td>Lavery Lighting &amp; Design:Store #JL-08</td>
      <td>Partial ship 12/02/2007</td>
      <td>-SPLIT-</td>
      <td>2,708.10</td>
      <td>0.00</td>
      <td>2,708.10</td>
    </tr>
    <tr>
      <th>607</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-12-07</td>
      <td>71138</td>
      <td>Baker's Professional Lighting:Store #10</td>
      <td>Partial ship 12/7/2007</td>
      <td>-SPLIT-</td>
      <td>10,680.48</td>
      <td>0.00</td>
      <td>10,680.48</td>
    </tr>
    <tr>
      <th>608</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-12-10</td>
      <td>71105</td>
      <td>Miscellaneous - Retail:Lara Gussman</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,679.00</td>
      <td>0.00</td>
      <td>1,679.00</td>
    </tr>
    <tr>
      <th>609</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-12-10</td>
      <td>66587</td>
      <td>Miscellaneous - Retail:Lara Gussman</td>
      <td>71105</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>1,679.00</td>
      <td>-1,679.00</td>
    </tr>
    <tr>
      <th>610</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-12-11</td>
      <td>5977</td>
      <td>Kern Lighting Warehouse:Store #01</td>
      <td>71133</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>9,839.00</td>
      <td>-9,839.00</td>
    </tr>
    <tr>
      <th>611</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-12-12</td>
      <td>71106</td>
      <td>Stern Commercial Contractor's:Walker Properties</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>21,330.00</td>
      <td>0.00</td>
      <td>21,330.00</td>
    </tr>
    <tr>
      <th>612</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-12-14</td>
      <td>5950</td>
      <td>Kern Lighting Warehouse:Store #34</td>
      <td>71132</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>10,723.60</td>
      <td>-10,723.60</td>
    </tr>
    <tr>
      <th>613</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-12-15</td>
      <td>71134</td>
      <td>Kern Lighting Warehouse:Store #13</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>9,033.50</td>
      <td>0.00</td>
      <td>9,033.50</td>
    </tr>
    <tr>
      <th>614</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-12-15</td>
      <td>71135</td>
      <td>Baker's Professional Lighting:Store #15</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,422.00</td>
      <td>0.00</td>
      <td>1,422.00</td>
    </tr>
    <tr>
      <th>615</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-12-15</td>
      <td>71136</td>
      <td>Kern Lighting Warehouse:Store #13</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>2,222.50</td>
      <td>0.00</td>
      <td>2,222.50</td>
    </tr>
    <tr>
      <th>616</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-12-15</td>
      <td>71137</td>
      <td>Dan A. North Builders:Custom Order - Suite 100A</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>3,500.00</td>
      <td>0.00</td>
      <td>3,500.00</td>
    </tr>
    <tr>
      <th>617</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-12-15</td>
      <td>71140</td>
      <td>Thompson Lighting Stores:Store #15</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>11,800.00</td>
      <td>0.00</td>
      <td>11,800.00</td>
    </tr>
    <tr>
      <th>618</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-12-15</td>
      <td>11301</td>
      <td>Lavery Lighting &amp; Design:Store #JL-01</td>
      <td>71117</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>5,279.00</td>
      <td>-5,279.00</td>
    </tr>
    <tr>
      <th>619</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-12-15</td>
      <td>5999</td>
      <td>Kern Lighting Warehouse:Store #13</td>
      <td>71134</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>9,033.50</td>
      <td>-9,033.50</td>
    </tr>
  </tbody>
</table>
<p>182 rows × 10 columns</p>
</div>



Okay, now this is important, I will further just investigate Accounts Receivable. Here you can see from the below plot the credit side of AR is slightly lower around the zero bound. This showed that there is slightly less amounts or repayment at that value. 


```python
ar["Debit"].sum()
```




    408309.53




```python
ar["Credit"].sum()
```




    375976.45



This company has clearly done well, there is a very small amoount of AR outstanding.


```python
%matplotlib inline
import matplotlib.pyplot as plt

ar.hist()
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x115d019b0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x115d3ad30>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x115d73c50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x115dabc50>]],
          dtype=object)




![png](General%20Ledger_files/General%20Ledger_26_1.png)


Here you can have a look at the values over the period, the numbers below is the daily values. It seems that towards the end of the period they are extending larger amounts of credit.


```python
ar.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a17affd68>




![png](General%20Ledger_files/General%20Ledger_28_1.png)


Now I will create a rolling commulative sum, doing this whe can see if there is any seasonality in the balance.


```python
ar["cum_net"] = ar["Net"].cumsum()
```

    /Users/dereksnow/anaconda/envs/py36/lib/python3.6/site-packages/ipykernel/__main__.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if __name__ == '__main__':


You can see that the summation remains within range, it is not to bad, there is a slight increase over the period. To prove there is an increase over time, we can run a linear regression to get the slope coefficient. 


```python
ar["Date"] =  pd.to_datetime(ar["Date"])
```

    /Users/dereksnow/anaconda/envs/py36/lib/python3.6/site-packages/ipykernel/__main__.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if __name__ == '__main__':



```python
ar = ar.set_index("Date",drop=True)
```


```python
ar["Credit"] = - ar["Credit"]
```


```python

```




    LinregressResult(slope=0.002061083196531608, intercept=32.73900646747864, rvalue=0.5445332646944847, pvalue=1.9277033049221335e-15, stderr=0.00023662564997062874)




```python
ar
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
      <th>Acct</th>
      <th>Type</th>
      <th>Num</th>
      <th>Name</th>
      <th>Memo</th>
      <th>Split</th>
      <th>Debit</th>
      <th>Credit</th>
      <th>Net</th>
      <th>cum_net</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-06</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71047</td>
      <td>Baker's Professional Lighting:Store #25</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>2,381.00</td>
      <td>-0.00</td>
      <td>2,381.00</td>
      <td>2,381.00</td>
    </tr>
    <tr>
      <th>2018-01-14</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71050</td>
      <td>Godwin Lighting Depot:Store #202</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>7,786.40</td>
      <td>-0.00</td>
      <td>7,786.40</td>
      <td>10,167.40</td>
    </tr>
    <tr>
      <th>2018-01-17</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71055</td>
      <td>Miscellaneous - Retail:Ms. Jann Minor</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,084.00</td>
      <td>-0.00</td>
      <td>1,084.00</td>
      <td>11,251.40</td>
    </tr>
    <tr>
      <th>2018-01-17</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>555</td>
      <td>Miscellaneous - Retail:Ms. Jann Minor</td>
      <td>71055</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-1,084.00</td>
      <td>-1,084.00</td>
      <td>10,167.40</td>
    </tr>
    <tr>
      <th>2018-01-18</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71092</td>
      <td>Miscellaneous - Retail:Brian Stern</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,126.00</td>
      <td>-0.00</td>
      <td>1,126.00</td>
      <td>11,293.40</td>
    </tr>
    <tr>
      <th>2018-01-18</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>11124</td>
      <td>Miscellaneous - Retail:Brian Stern</td>
      <td>71092</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-1,126.00</td>
      <td>-1,126.00</td>
      <td>10,167.40</td>
    </tr>
    <tr>
      <th>2018-01-28</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71053</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>854.00</td>
      <td>-0.00</td>
      <td>854.00</td>
      <td>11,021.40</td>
    </tr>
    <tr>
      <th>2018-01-28</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71072</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>2,187.45</td>
      <td>-0.00</td>
      <td>2,187.45</td>
      <td>13,208.85</td>
    </tr>
    <tr>
      <th>2018-01-28</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>254</td>
      <td>Miscellaneous - Retail:Alison Johnson</td>
      <td>71053</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-854.00</td>
      <td>-854.00</td>
      <td>12,354.85</td>
    </tr>
    <tr>
      <th>2018-01-29</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71124</td>
      <td>Kern Lighting Warehouse:Store #34</td>
      <td>NA</td>
      <td>Revenue</td>
      <td>8,400.00</td>
      <td>-0.00</td>
      <td>8,400.00</td>
      <td>20,754.85</td>
    </tr>
    <tr>
      <th>2018-01-31</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71059</td>
      <td>Godwin Lighting Depot:Store #303</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,687.95</td>
      <td>-0.00</td>
      <td>1,687.95</td>
      <td>22,442.80</td>
    </tr>
    <tr>
      <th>2018-02-01</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71121</td>
      <td>Kern Lighting Warehouse:Store #34</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>6,745.00</td>
      <td>-0.00</td>
      <td>6,745.00</td>
      <td>29,187.80</td>
    </tr>
    <tr>
      <th>2018-02-09</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>130</td>
      <td>Baker's Professional Lighting:Store #25</td>
      <td>71047</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-2,381.00</td>
      <td>-2,381.00</td>
      <td>26,806.80</td>
    </tr>
    <tr>
      <th>2018-02-10</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71051</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>4,364.00</td>
      <td>-0.00</td>
      <td>4,364.00</td>
      <td>31,170.80</td>
    </tr>
    <tr>
      <th>2018-02-11</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71112</td>
      <td>Thompson Lighting Stores:Store #15</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>3,025.00</td>
      <td>-0.00</td>
      <td>3,025.00</td>
      <td>34,195.80</td>
    </tr>
    <tr>
      <th>2018-02-12</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71088</td>
      <td>Cole Home Builders:Phase 1 - Lot 2</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>2,187.45</td>
      <td>-0.00</td>
      <td>2,187.45</td>
      <td>36,383.25</td>
    </tr>
    <tr>
      <th>2018-02-14</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71110</td>
      <td>Baker's Professional Lighting:Store #05</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>2,391.00</td>
      <td>-0.00</td>
      <td>2,391.00</td>
      <td>38,774.25</td>
    </tr>
    <tr>
      <th>2018-02-15</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>1103</td>
      <td>Stern Commercial Contractor's:Walker Properties</td>
      <td>71106</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-21,330.00</td>
      <td>-21,330.00</td>
      <td>17,444.25</td>
    </tr>
    <tr>
      <th>2018-02-18</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71093</td>
      <td>Miscellaneous - Retail:Ruth Kuver</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,366.00</td>
      <td>-0.00</td>
      <td>1,366.00</td>
      <td>18,810.25</td>
    </tr>
    <tr>
      <th>2018-02-18</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>5467</td>
      <td>Miscellaneous - Retail:Ruth Kuver</td>
      <td>71093</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-1,366.00</td>
      <td>-1,366.00</td>
      <td>17,444.25</td>
    </tr>
    <tr>
      <th>2018-02-20</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71054</td>
      <td>Miscellaneous - Retail:Mrs. Chris Holly</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,300.00</td>
      <td>-0.00</td>
      <td>1,300.00</td>
      <td>18,744.25</td>
    </tr>
    <tr>
      <th>2018-02-20</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>305</td>
      <td>Miscellaneous - Retail:Mrs. Chris Holly</td>
      <td>71054</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-1,300.00</td>
      <td>-1,300.00</td>
      <td>17,444.25</td>
    </tr>
    <tr>
      <th>2018-02-23</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71094</td>
      <td>Miscellaneous - Retail:David Lo</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,932.00</td>
      <td>-0.00</td>
      <td>1,932.00</td>
      <td>19,376.25</td>
    </tr>
    <tr>
      <th>2018-02-23</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>12357</td>
      <td>Miscellaneous - Retail:David Lo</td>
      <td>71094</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-1,932.00</td>
      <td>-1,932.00</td>
      <td>17,444.25</td>
    </tr>
    <tr>
      <th>2018-02-27</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71052</td>
      <td>Baker's Professional Lighting:Store #15</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,885.00</td>
      <td>-0.00</td>
      <td>1,885.00</td>
      <td>19,329.25</td>
    </tr>
    <tr>
      <th>2018-02-28</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71122</td>
      <td>Lavery Lighting &amp; Design:Store #JL-08</td>
      <td>NA</td>
      <td>Revenue</td>
      <td>600.00</td>
      <td>-0.00</td>
      <td>600.00</td>
      <td>19,929.25</td>
    </tr>
    <tr>
      <th>2018-02-28</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>57702</td>
      <td>Godwin Lighting Depot:Store #202</td>
      <td>71050</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-7,786.40</td>
      <td>-7,786.40</td>
      <td>12,142.85</td>
    </tr>
    <tr>
      <th>2018-02-28</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>57713</td>
      <td>Godwin Lighting Depot:Store #303</td>
      <td>71059</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-1,687.95</td>
      <td>-1,687.95</td>
      <td>10,454.90</td>
    </tr>
    <tr>
      <th>2018-03-02</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>5749</td>
      <td>Kern Lighting Warehouse:Store #34</td>
      <td>71124</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-8,400.00</td>
      <td>-8,400.00</td>
      <td>2,054.90</td>
    </tr>
    <tr>
      <th>2018-03-02</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>5750</td>
      <td>Kern Lighting Warehouse:Store #13</td>
      <td>71123</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-110.00</td>
      <td>-110.00</td>
      <td>1,944.90</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-11-05</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71117</td>
      <td>Lavery Lighting &amp; Design:Store #JL-01</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>5,279.00</td>
      <td>-0.00</td>
      <td>5,279.00</td>
      <td>51,058.20</td>
    </tr>
    <tr>
      <th>2018-11-05</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>101</td>
      <td>Miscellaneous - Retail:Valesha Jones</td>
      <td>71103</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-2,395.00</td>
      <td>-2,395.00</td>
      <td>48,663.20</td>
    </tr>
    <tr>
      <th>2018-11-16</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71118</td>
      <td>Lavery Lighting &amp; Design:Store #JL-04</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>11,715.00</td>
      <td>-0.00</td>
      <td>11,715.00</td>
      <td>60,378.20</td>
    </tr>
    <tr>
      <th>2018-11-16</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>11270</td>
      <td>Cole Home Builders:Phase 1 - Lot 5</td>
      <td>71073</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-2,138.40</td>
      <td>-2,138.40</td>
      <td>58,239.80</td>
    </tr>
    <tr>
      <th>2018-11-16</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>11257</td>
      <td>Cole Home Builders:Phase 1 - Lot 5</td>
      <td>71074</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-2,138.40</td>
      <td>-2,138.40</td>
      <td>56,101.40</td>
    </tr>
    <tr>
      <th>2018-11-16</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>11250</td>
      <td>Cole Home Builders:Phase 1 - Lot 5</td>
      <td>71075</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-2,138.40</td>
      <td>-2,138.40</td>
      <td>53,963.00</td>
    </tr>
    <tr>
      <th>2018-11-16</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>11280</td>
      <td>Cole Home Builders:Phase 2 - Lot 5</td>
      <td>71091</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-2,138.40</td>
      <td>-2,138.40</td>
      <td>51,824.60</td>
    </tr>
    <tr>
      <th>2018-11-16</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>5930</td>
      <td>Kern Lighting Warehouse:Store #13</td>
      <td>71131</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-5,569.00</td>
      <td>-5,569.00</td>
      <td>46,255.60</td>
    </tr>
    <tr>
      <th>2018-11-20</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71104</td>
      <td>Miscellaneous - Retail:Melanie Hall</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,600.00</td>
      <td>-0.00</td>
      <td>1,600.00</td>
      <td>47,855.60</td>
    </tr>
    <tr>
      <th>2018-11-20</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>55449</td>
      <td>Miscellaneous - Retail:Melanie Hall</td>
      <td>71104</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-1,600.00</td>
      <td>-1,600.00</td>
      <td>46,255.60</td>
    </tr>
    <tr>
      <th>2018-11-22</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71133</td>
      <td>Kern Lighting Warehouse:Store #01</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>9,839.00</td>
      <td>-0.00</td>
      <td>9,839.00</td>
      <td>56,094.60</td>
    </tr>
    <tr>
      <th>2018-11-29</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71123</td>
      <td>Kern Lighting Warehouse:Store #13</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>110.00</td>
      <td>-0.00</td>
      <td>110.00</td>
      <td>56,204.60</td>
    </tr>
    <tr>
      <th>2018-11-29</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>1052</td>
      <td>Stern Commercial Contractor's:Tittle Properties</td>
      <td>71107</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-17,433.00</td>
      <td>-17,433.00</td>
      <td>38,771.60</td>
    </tr>
    <tr>
      <th>2018-11-30</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>5791</td>
      <td>Kern Lighting Warehouse:Store #13</td>
      <td>71120</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-8,190.00</td>
      <td>-8,190.00</td>
      <td>30,581.60</td>
    </tr>
    <tr>
      <th>2018-11-30</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>1098</td>
      <td>Stern Commercial Contractor's:Wilson Suites</td>
      <td>71108</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-14,355.00</td>
      <td>-14,355.00</td>
      <td>16,226.60</td>
    </tr>
    <tr>
      <th>2018-12-01</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>11311</td>
      <td>Lavery Lighting &amp; Design:Store #JL-04</td>
      <td>71118</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-11,715.00</td>
      <td>-11,715.00</td>
      <td>4,511.60</td>
    </tr>
    <tr>
      <th>2018-12-03</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71139</td>
      <td>Lavery Lighting &amp; Design:Store #JL-08</td>
      <td>Partial ship 12/02/2007</td>
      <td>-SPLIT-</td>
      <td>2,708.10</td>
      <td>-0.00</td>
      <td>2,708.10</td>
      <td>7,219.70</td>
    </tr>
    <tr>
      <th>2018-12-07</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71138</td>
      <td>Baker's Professional Lighting:Store #10</td>
      <td>Partial ship 12/7/2007</td>
      <td>-SPLIT-</td>
      <td>10,680.48</td>
      <td>-0.00</td>
      <td>10,680.48</td>
      <td>17,900.18</td>
    </tr>
    <tr>
      <th>2018-12-10</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71105</td>
      <td>Miscellaneous - Retail:Lara Gussman</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,679.00</td>
      <td>-0.00</td>
      <td>1,679.00</td>
      <td>19,579.18</td>
    </tr>
    <tr>
      <th>2018-12-10</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>66587</td>
      <td>Miscellaneous - Retail:Lara Gussman</td>
      <td>71105</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-1,679.00</td>
      <td>-1,679.00</td>
      <td>17,900.18</td>
    </tr>
    <tr>
      <th>2018-12-11</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>5977</td>
      <td>Kern Lighting Warehouse:Store #01</td>
      <td>71133</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-9,839.00</td>
      <td>-9,839.00</td>
      <td>8,061.18</td>
    </tr>
    <tr>
      <th>2018-12-12</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71106</td>
      <td>Stern Commercial Contractor's:Walker Properties</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>21,330.00</td>
      <td>-0.00</td>
      <td>21,330.00</td>
      <td>29,391.18</td>
    </tr>
    <tr>
      <th>2018-12-14</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>5950</td>
      <td>Kern Lighting Warehouse:Store #34</td>
      <td>71132</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-10,723.60</td>
      <td>-10,723.60</td>
      <td>18,667.58</td>
    </tr>
    <tr>
      <th>2018-12-15</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71134</td>
      <td>Kern Lighting Warehouse:Store #13</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>9,033.50</td>
      <td>-0.00</td>
      <td>9,033.50</td>
      <td>27,701.08</td>
    </tr>
    <tr>
      <th>2018-12-15</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71135</td>
      <td>Baker's Professional Lighting:Store #15</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,422.00</td>
      <td>-0.00</td>
      <td>1,422.00</td>
      <td>29,123.08</td>
    </tr>
    <tr>
      <th>2018-12-15</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71136</td>
      <td>Kern Lighting Warehouse:Store #13</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>2,222.50</td>
      <td>-0.00</td>
      <td>2,222.50</td>
      <td>31,345.58</td>
    </tr>
    <tr>
      <th>2018-12-15</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71137</td>
      <td>Dan A. North Builders:Custom Order - Suite 100A</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>3,500.00</td>
      <td>-0.00</td>
      <td>3,500.00</td>
      <td>34,845.58</td>
    </tr>
    <tr>
      <th>2018-12-15</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71140</td>
      <td>Thompson Lighting Stores:Store #15</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>11,800.00</td>
      <td>-0.00</td>
      <td>11,800.00</td>
      <td>46,645.58</td>
    </tr>
    <tr>
      <th>2018-12-15</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>11301</td>
      <td>Lavery Lighting &amp; Design:Store #JL-01</td>
      <td>71117</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-5,279.00</td>
      <td>-5,279.00</td>
      <td>41,366.58</td>
    </tr>
    <tr>
      <th>2018-12-15</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>5999</td>
      <td>Kern Lighting Warehouse:Store #13</td>
      <td>71134</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-9,033.50</td>
      <td>-9,033.50</td>
      <td>32,333.08</td>
    </tr>
  </tbody>
</table>
<p>182 rows × 10 columns</p>
</div>




```python
ar["cum_net"].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x115b63358>




![png](General%20Ledger_files/General%20Ledger_37_1.png)



```python
ar.head()
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
      <th>Acct</th>
      <th>Type</th>
      <th>Date</th>
      <th>Num</th>
      <th>Name</th>
      <th>Memo</th>
      <th>Split</th>
      <th>Debit</th>
      <th>Credit</th>
      <th>Net</th>
      <th>cum_net</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>438</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-01-06</td>
      <td>71047</td>
      <td>Baker's Professional Lighting:Store #25</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>2,381.00</td>
      <td>0.00</td>
      <td>2,381.00</td>
      <td>2,381.00</td>
    </tr>
    <tr>
      <th>439</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-01-14</td>
      <td>71050</td>
      <td>Godwin Lighting Depot:Store #202</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>7,786.40</td>
      <td>0.00</td>
      <td>7,786.40</td>
      <td>10,167.40</td>
    </tr>
    <tr>
      <th>440</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-01-17</td>
      <td>71055</td>
      <td>Miscellaneous - Retail:Ms. Jann Minor</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,084.00</td>
      <td>0.00</td>
      <td>1,084.00</td>
      <td>11,251.40</td>
    </tr>
    <tr>
      <th>441</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>2018-01-17</td>
      <td>555</td>
      <td>Miscellaneous - Retail:Ms. Jann Minor</td>
      <td>71055</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>1,084.00</td>
      <td>-1,084.00</td>
      <td>10,167.40</td>
    </tr>
    <tr>
      <th>442</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>2018-01-18</td>
      <td>71092</td>
      <td>Miscellaneous - Retail:Brian Stern</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,126.00</td>
      <td>0.00</td>
      <td>1,126.00</td>
      <td>11,293.40</td>
    </tr>
  </tbody>
</table>
</div>




```python
ar.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a17b85f28>




![png](General%20Ledger_files/General%20Ledger_39_1.png)



```python
from scipy.stats import linregress
linregress(list(ar["cum_net"].values), list(range(len(ar))))

```




    LinregressResult(slope=0.002061083196531608, intercept=32.73900646747864, rvalue=0.5445332646944847, pvalue=1.9277033049221335e-15, stderr=0.00023662564997062874)




```python
ar.head()
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
      <th>Acct</th>
      <th>Type</th>
      <th>Num</th>
      <th>Name</th>
      <th>Memo</th>
      <th>Split</th>
      <th>Debit</th>
      <th>Credit</th>
      <th>Net</th>
      <th>cum_net</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-06</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71047</td>
      <td>Baker's Professional Lighting:Store #25</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>2,381.00</td>
      <td>-0.00</td>
      <td>2,381.00</td>
      <td>2,381.00</td>
    </tr>
    <tr>
      <th>2018-01-14</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71050</td>
      <td>Godwin Lighting Depot:Store #202</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>7,786.40</td>
      <td>-0.00</td>
      <td>7,786.40</td>
      <td>10,167.40</td>
    </tr>
    <tr>
      <th>2018-01-17</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71055</td>
      <td>Miscellaneous - Retail:Ms. Jann Minor</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,084.00</td>
      <td>-0.00</td>
      <td>1,084.00</td>
      <td>11,251.40</td>
    </tr>
    <tr>
      <th>2018-01-17</th>
      <td>Accounts Receivable</td>
      <td>Payment</td>
      <td>555</td>
      <td>Miscellaneous - Retail:Ms. Jann Minor</td>
      <td>71055</td>
      <td>Undeposited Funds</td>
      <td>0.00</td>
      <td>-1,084.00</td>
      <td>-1,084.00</td>
      <td>10,167.40</td>
    </tr>
    <tr>
      <th>2018-01-18</th>
      <td>Accounts Receivable</td>
      <td>Invoice</td>
      <td>71092</td>
      <td>Miscellaneous - Retail:Brian Stern</td>
      <td>NA</td>
      <td>-SPLIT-</td>
      <td>1,126.00</td>
      <td>-0.00</td>
      <td>1,126.00</td>
      <td>11,293.40</td>
    </tr>
  </tbody>
</table>
</div>




```python
from wordcloud import WordCloud

wordcloud2 = WordCloud().generate(' '.join(ar['Name']))
```


```python
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()
```


![png](General%20Ledger_files/General%20Ledger_43_0.png)



```python
### Here you can focus more on the debtors and look at the debtors aged report.
## https://www.accountingweb.co.uk/tech/tech-pulse/coding-for-accountants-creating-an-aged-debtors-report
```
