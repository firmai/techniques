
# RFM Analysis

This iPython notebook explains how to perform RFM analysis from customer purchase history data. The sample orders file is Sample - Superstore dataset from Tableau Software.



```python
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
```

Read the sample orders file, containing all past purchases for all customers.


```python
orders = pd.read_csv('sample-orders2.csv',sep=',')
```


```python
orders.head()
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
      <th>order_date</th>
      <th>order_id</th>
      <th>customer</th>
      <th>grand_total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9/07/11</td>
      <td>CA-2011-100006</td>
      <td>Dennis Kane</td>
      <td>378</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7/08/11</td>
      <td>CA-2011-100090</td>
      <td>Ed Braxton</td>
      <td>699</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3/14/11</td>
      <td>CA-2011-100293</td>
      <td>Neil Französisch</td>
      <td>91</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1/29/11</td>
      <td>CA-2011-100328</td>
      <td>Jasper Cacioppo</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4/08/11</td>
      <td>CA-2011-100363</td>
      <td>Jim Mitchum</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>



## Create the RFM Table

Since recency is calculated for a point in time and the Tableau Super Store dataset last order date is Dec 31 2014, that is the date we will use to calculate recency.

Set this date to the current day and extract all orders until yesterday.


```python
import datetime as dt
NOW = dt.datetime(2014,12,31)
```


```python
# Make the date_placed column datetime
orders['order_date'] = pd.to_datetime(orders['order_date'])
```

Create the RFM Table


```python
rfmTable = orders.groupby('customer').agg({'order_date': lambda x: (NOW - x.max()).days, # Recency
                                        'order_id': lambda x: len(x),      # Frequency
                                        'grand_total': lambda x: x.sum()}) # Monetary Value

rfmTable['order_date'] = rfmTable['order_date'].astype(int)
rfmTable.rename(columns={'order_date': 'recency', 
                         'order_id': 'frequency', 
                         'grand_total': 'monetary_value'}, inplace=True)
```

## Validating the RFM Table


```python
rfmTable.head()
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
      <th>recency</th>
      <th>frequency</th>
      <th>monetary_value</th>
    </tr>
    <tr>
      <th>customer</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aaron Bergman</th>
      <td>415</td>
      <td>3</td>
      <td>887</td>
    </tr>
    <tr>
      <th>Aaron Hawkins</th>
      <td>12</td>
      <td>7</td>
      <td>1744</td>
    </tr>
    <tr>
      <th>Aaron Smayling</th>
      <td>88</td>
      <td>7</td>
      <td>3050</td>
    </tr>
    <tr>
      <th>Adam Bellavance</th>
      <td>54</td>
      <td>8</td>
      <td>7756</td>
    </tr>
    <tr>
      <th>Adam Hart</th>
      <td>34</td>
      <td>10</td>
      <td>3249</td>
    </tr>
  </tbody>
</table>
</div>



Customer **Aaron Bergman** has frequency:3, monetary value:$887 and recency:415 days.


```python
aaron = orders[orders['customer']=='Aaron Bergman']
aaron
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
      <th>order_date</th>
      <th>order_id</th>
      <th>customer</th>
      <th>grand_total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>624</th>
      <td>2011-02-19</td>
      <td>CA-2011-152905</td>
      <td>Aaron Bergman</td>
      <td>13</td>
    </tr>
    <tr>
      <th>665</th>
      <td>2011-03-07</td>
      <td>CA-2011-156587</td>
      <td>Aaron Bergman</td>
      <td>310</td>
    </tr>
    <tr>
      <th>2336</th>
      <td>2013-11-11</td>
      <td>CA-2013-140935</td>
      <td>Aaron Bergman</td>
      <td>564</td>
    </tr>
  </tbody>
</table>
</div>



Inserting the date of Aaron purchase and comparing it to the recency in the rfmTable we verify our RFM table is correct.


```python
(NOW - dt.datetime(2013,11,11)).days==415
```




    True



## Determining RFM Quartiles


```python
quantiles = rfmTable.quantile(q=[0.25,0.5,0.75])
```


```python
#25th, 50th, 75th percentile.
quantiles
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
      <th>recency</th>
      <th>frequency</th>
      <th>monetary_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.25</th>
      <td>30.0</td>
      <td>5.0</td>
      <td>1145.0</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>75.0</td>
      <td>6.0</td>
      <td>2257.0</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>183.0</td>
      <td>8.0</td>
      <td>3784.0</td>
    </tr>
  </tbody>
</table>
</div>



Send quantiles to a dictionary, easier to use.


```python
quantiles = quantiles.to_dict()
```


```python
quantiles
```




    {'frequency': {0.25: 5.0, 0.5: 6.0, 0.75: 8.0},
     'monetary_value': {0.25: 1145.0, 0.5: 2257.0, 0.75: 3784.0},
     'recency': {0.25: 30.0, 0.5: 75.0, 0.75: 183.0}}



## Creating the RFM segmentation table


```python
rfmSegmentation = rfmTable
```

We create two classes for the RFM segmentation since, being high recency is bad, while high frequency and monetary value is good. 


```python
# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
def RClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
def FMClass(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1

```


```python
rfmSegmentation['R_Quartile'] = rfmSegmentation['recency'].apply(RClass, args=('recency',quantiles,))
rfmSegmentation['F_Quartile'] = rfmSegmentation['frequency'].apply(FMClass, args=('frequency',quantiles,))
rfmSegmentation['M_Quartile'] = rfmSegmentation['monetary_value'].apply(FMClass, args=('monetary_value',quantiles,))
```


```python
rfmSegmentation['RFMClass'] = rfmSegmentation.R_Quartile.map(str) \
                            + rfmSegmentation.F_Quartile.map(str) \
                            + rfmSegmentation.M_Quartile.map(str)
```


```python
rfmSegmentation.head()
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
      <th>recency</th>
      <th>frequency</th>
      <th>monetary_value</th>
      <th>R_Quartile</th>
      <th>F_Quartile</th>
      <th>M_Quartile</th>
      <th>RFMClass</th>
    </tr>
    <tr>
      <th>customer</th>
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
      <th>Aaron Bergman</th>
      <td>415</td>
      <td>3</td>
      <td>887</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>444</td>
    </tr>
    <tr>
      <th>Aaron Hawkins</th>
      <td>12</td>
      <td>7</td>
      <td>1744</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>123</td>
    </tr>
    <tr>
      <th>Aaron Smayling</th>
      <td>88</td>
      <td>7</td>
      <td>3050</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>322</td>
    </tr>
    <tr>
      <th>Adam Bellavance</th>
      <td>54</td>
      <td>8</td>
      <td>7756</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>221</td>
    </tr>
    <tr>
      <th>Adam Hart</th>
      <td>34</td>
      <td>10</td>
      <td>3249</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>212</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Uncomment any of the following lines to: copy data to clipboard or save it to a CSV file.
# rfmSegmentation.to_clipboard()
# rfmSegmentation.to_csv('rfm-table.csv', sep=',')
```

Who are the top 5 best customers? by RFM Class (111), high spenders who buy recently and frequently?


```python
rfmSegmentation[rfmSegmentation['RFMClass']=='111'].sort_values('monetary_value', ascending=False).head(5)
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
      <th>recency</th>
      <th>frequency</th>
      <th>monetary_value</th>
      <th>R_Quartile</th>
      <th>F_Quartile</th>
      <th>M_Quartile</th>
      <th>RFMClass</th>
    </tr>
    <tr>
      <th>customer</th>
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
      <th>Sanjit Engle</th>
      <td>9</td>
      <td>11</td>
      <td>12210</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>111</td>
    </tr>
    <tr>
      <th>John Lee</th>
      <td>21</td>
      <td>11</td>
      <td>9801</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>111</td>
    </tr>
    <tr>
      <th>Pete Kriz</th>
      <td>9</td>
      <td>12</td>
      <td>8647</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>111</td>
    </tr>
    <tr>
      <th>Harry Marie</th>
      <td>2</td>
      <td>10</td>
      <td>8237</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>111</td>
    </tr>
    <tr>
      <th>Lena Creighton</th>
      <td>16</td>
      <td>12</td>
      <td>7661</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>111</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Question, can the RFM class be automated using KNN - What clusters does it find. 
```
