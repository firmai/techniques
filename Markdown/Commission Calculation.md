
# Introduction

# Collecting the Data

Import pandas and numpy


```python
import pandas as pd
import numpy as np
```

Let's take a look at the files in our input directory, using the convenient shell commands in ipython.


```python
!ls data
```

    [31m311-service-requests.csv[m[m
    [31mAll-Web-Site-Data-Audience-Overview.xlsx[m[m
    [31mAussie_Wines_Plotting.csv[m[m
    [31mMarch-2017-forecast-article.xlsx[m[m
    [31mTraffic_20170306-20170519.xlsx[m[m
    [31mTweets.csv[m[m
    [31mairbnb_session_data.txt[m[m
    [31mcustomer-status.xlsx[m[m
    [31mdebtors.csv[m[m
    [31mexcel_sample.xlsx[m[m
    [31mgeneral-ledger-sample.xlsx[m[m
    [31mmn-budget-detail-2014.csv[m[m
    [31mpnl.xlsx[m[m
    [31mpopulation.xlsx[m[m
    [31msales-estimate.xlsx[m[m
    [31msales-feb-2014.xlsx[m[m
    [31msales-jan-2014.xlsx[m[m
    [31msales-mar-2014.xlsx[m[m
    [31msales_data_types.csv[m[m
    [31msales_transactions.xlsx[m[m
    [31msalesfunnel.xlsx[m[m
    [31msample-sales-reps.xlsx[m[m
    [31msample-sales-tax.csv[m[m
    [31msample-salesv3.xlsx[m[m
    [31mtb_import.xlsx[m[m
    [31m~$general-ledger-sample.xlsx[m[m


There are a lot of files, but we only want to look at the sales .xlsx files.


```python
!ls data/sales-*-2014.xlsx
```

    [31mdata/sales-feb-2014.xlsx[m[m [31mdata/sales-jan-2014.xlsx[m[m [31mdata/sales-mar-2014.xlsx[m[m


Use the python glob module to easily list out the files we need


```python
import glob
```


```python
glob.glob("data/sales-*-2014.xlsx")
```




    ['data/sales-feb-2014.xlsx',
     'data/sales-jan-2014.xlsx',
     'data/sales-mar-2014.xlsx']



This gives us what we need, let's import each of our files and combine them into one file. 

Panda's concat and append can do this for us. I'm going to use append in this example.

The code snippet below will initialize a blank DataFrame then append all of the individual files into the all_data DataFrame.


```python
all_data = pd.DataFrame()
for f in glob.glob("data/sales-*-2014.xlsx"):
    df = pd.read_excel(f)
    all_data = all_data.append(df,ignore_index=True)
```

Now we have all the data in our all_data DataFrame. You can use describe to look at it and make sure you data looks good.


```python
all_data.shape
```




    (384, 7)




```python
all_data.describe()
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
      <th>account number</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>384.000000</td>
      <td>384.000000</td>
      <td>384.000000</td>
      <td>384.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>478125.989583</td>
      <td>24.372396</td>
      <td>56.651406</td>
      <td>1394.517344</td>
    </tr>
    <tr>
      <th>std</th>
      <td>220902.947401</td>
      <td>14.373219</td>
      <td>27.075883</td>
      <td>1117.809743</td>
    </tr>
    <tr>
      <th>min</th>
      <td>141962.000000</td>
      <td>-1.000000</td>
      <td>10.210000</td>
      <td>-97.160000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>257198.000000</td>
      <td>12.000000</td>
      <td>32.612500</td>
      <td>482.745000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>424914.000000</td>
      <td>23.500000</td>
      <td>58.160000</td>
      <td>1098.710000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>714466.000000</td>
      <td>37.000000</td>
      <td>80.965000</td>
      <td>2132.260000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>786968.000000</td>
      <td>49.000000</td>
      <td>99.730000</td>
      <td>4590.810000</td>
    </tr>
  </tbody>
</table>
</div>



Alot of this data may not make much sense for this data set but I'm most interested in the count row to make sure the number of data elements makes sense.


```python
all_data.head()
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
      <th>account number</th>
      <th>name</th>
      <th>sku</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>383080</td>
      <td>Will LLC</td>
      <td>B1-20000</td>
      <td>7</td>
      <td>33.69</td>
      <td>235.83</td>
      <td>2014-02-01 09:04:59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>S1-27722</td>
      <td>11</td>
      <td>21.12</td>
      <td>232.32</td>
      <td>2014-02-01 11:51:46</td>
    </tr>
    <tr>
      <th>2</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>B1-86481</td>
      <td>3</td>
      <td>35.99</td>
      <td>107.97</td>
      <td>2014-02-01 17:24:32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>B1-20000</td>
      <td>23</td>
      <td>78.90</td>
      <td>1814.70</td>
      <td>2014-02-01 19:56:48</td>
    </tr>
    <tr>
      <th>4</th>
      <td>672390</td>
      <td>Kuhn-Gusikowski</td>
      <td>S1-06532</td>
      <td>48</td>
      <td>55.82</td>
      <td>2679.36</td>
      <td>2014-02-02 03:45:20</td>
    </tr>
  </tbody>
</table>
</div>



It is not critical in this example but the best practice is to convert the date column to a date time object.


```python
all_data['date'] = pd.to_datetime(all_data['date'])
```

# Combining Data

Now that we have all of the data into one DataFrame, we can do any manipulations the DataFrame supports. In this case, the next thing we want to do is read in another file that contains the customer status by account. You can think of this as a company's customer segmentation strategy or some other mechanism for identifying their customers.

First, we read in the data.


```python
status = pd.read_excel("data/customer-status.xlsx")
status
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
      <th>account number</th>
      <th>name</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>740150</td>
      <td>Barton LLC</td>
      <td>gold</td>
    </tr>
    <tr>
      <th>1</th>
      <td>714466</td>
      <td>Trantow-Barrows</td>
      <td>silver</td>
    </tr>
    <tr>
      <th>2</th>
      <td>218895</td>
      <td>Kulas Inc</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>3</th>
      <td>307599</td>
      <td>Kassulke, Ondricka and Metz</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>4</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>5</th>
      <td>729833</td>
      <td>Koepp Ltd</td>
      <td>silver</td>
    </tr>
    <tr>
      <th>6</th>
      <td>146832</td>
      <td>Kiehn-Spinka</td>
      <td>silver</td>
    </tr>
    <tr>
      <th>7</th>
      <td>688981</td>
      <td>Keeling LLC</td>
      <td>silver</td>
    </tr>
    <tr>
      <th>8</th>
      <td>786968</td>
      <td>Frami, Hills and Schmidt</td>
      <td>silver</td>
    </tr>
    <tr>
      <th>9</th>
      <td>239344</td>
      <td>Stokes LLC</td>
      <td>gold</td>
    </tr>
    <tr>
      <th>10</th>
      <td>672390</td>
      <td>Kuhn-Gusikowski</td>
      <td>silver</td>
    </tr>
    <tr>
      <th>11</th>
      <td>141962</td>
      <td>Herman LLC</td>
      <td>gold</td>
    </tr>
    <tr>
      <th>12</th>
      <td>424914</td>
      <td>White-Trantow</td>
      <td>silver</td>
    </tr>
    <tr>
      <th>13</th>
      <td>527099</td>
      <td>Sanford and Sons</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>14</th>
      <td>642753</td>
      <td>Pollich LLC</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>15</th>
      <td>257198</td>
      <td>Cronin, Oberbrunner and Spencer</td>
      <td>gold</td>
    </tr>
  </tbody>
</table>
</div>



We want to merge this data with our concatenated data set of sales. We use panda's merge function and tell it to do a left join which is similar to Excel's vlookup function.


```python
all_data_st = pd.merge(all_data, status, how='left')
all_data_st.head()
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
      <th>account number</th>
      <th>name</th>
      <th>sku</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>383080</td>
      <td>Will LLC</td>
      <td>B1-20000</td>
      <td>7</td>
      <td>33.69</td>
      <td>235.83</td>
      <td>2014-02-01 09:04:59</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>S1-27722</td>
      <td>11</td>
      <td>21.12</td>
      <td>232.32</td>
      <td>2014-02-01 11:51:46</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>2</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>B1-86481</td>
      <td>3</td>
      <td>35.99</td>
      <td>107.97</td>
      <td>2014-02-01 17:24:32</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>3</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>B1-20000</td>
      <td>23</td>
      <td>78.90</td>
      <td>1814.70</td>
      <td>2014-02-01 19:56:48</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>4</th>
      <td>672390</td>
      <td>Kuhn-Gusikowski</td>
      <td>S1-06532</td>
      <td>48</td>
      <td>55.82</td>
      <td>2679.36</td>
      <td>2014-02-02 03:45:20</td>
      <td>silver</td>
    </tr>
  </tbody>
</table>
</div>



This looks pretty good but let's look at a specific account.


```python
all_data_st[all_data_st["account number"]==737550].head()
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
      <th>account number</th>
      <th>name</th>
      <th>sku</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>737550</td>
      <td>Fritsch, Russel and Anderson</td>
      <td>S1-47412</td>
      <td>40</td>
      <td>51.01</td>
      <td>2040.40</td>
      <td>2014-02-05 01:20:40</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>737550</td>
      <td>Fritsch, Russel and Anderson</td>
      <td>S1-06532</td>
      <td>34</td>
      <td>18.69</td>
      <td>635.46</td>
      <td>2014-02-07 09:22:02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>66</th>
      <td>737550</td>
      <td>Fritsch, Russel and Anderson</td>
      <td>S1-27722</td>
      <td>15</td>
      <td>70.23</td>
      <td>1053.45</td>
      <td>2014-02-16 18:24:42</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>78</th>
      <td>737550</td>
      <td>Fritsch, Russel and Anderson</td>
      <td>S2-34077</td>
      <td>26</td>
      <td>93.35</td>
      <td>2427.10</td>
      <td>2014-02-20 18:45:43</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>80</th>
      <td>737550</td>
      <td>Fritsch, Russel and Anderson</td>
      <td>S1-93683</td>
      <td>31</td>
      <td>10.52</td>
      <td>326.12</td>
      <td>2014-02-21 13:55:45</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



This account number was not in our status file, so we have a bunch of NaN's. We can decide how we want to handle this situation. For this specific case, let's label all missing accounts as bronze. Use the fillna function to easily accomplish this on the status column.


```python
all_data_st['status'].fillna('bronze',inplace=True)
all_data_st.head()
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
      <th>account number</th>
      <th>name</th>
      <th>sku</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>383080</td>
      <td>Will LLC</td>
      <td>B1-20000</td>
      <td>7</td>
      <td>33.69</td>
      <td>235.83</td>
      <td>2014-02-01 09:04:59</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>1</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>S1-27722</td>
      <td>11</td>
      <td>21.12</td>
      <td>232.32</td>
      <td>2014-02-01 11:51:46</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>2</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>B1-86481</td>
      <td>3</td>
      <td>35.99</td>
      <td>107.97</td>
      <td>2014-02-01 17:24:32</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>3</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>B1-20000</td>
      <td>23</td>
      <td>78.90</td>
      <td>1814.70</td>
      <td>2014-02-01 19:56:48</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>4</th>
      <td>672390</td>
      <td>Kuhn-Gusikowski</td>
      <td>S1-06532</td>
      <td>48</td>
      <td>55.82</td>
      <td>2679.36</td>
      <td>2014-02-02 03:45:20</td>
      <td>silver</td>
    </tr>
  </tbody>
</table>
</div>



Check the data just to make sure we're all good.


```python
all_data_st[all_data_st["account number"]==737550].head()
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
      <th>account number</th>
      <th>name</th>
      <th>sku</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>737550</td>
      <td>Fritsch, Russel and Anderson</td>
      <td>S1-47412</td>
      <td>40</td>
      <td>51.01</td>
      <td>2040.40</td>
      <td>2014-02-05 01:20:40</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>25</th>
      <td>737550</td>
      <td>Fritsch, Russel and Anderson</td>
      <td>S1-06532</td>
      <td>34</td>
      <td>18.69</td>
      <td>635.46</td>
      <td>2014-02-07 09:22:02</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>66</th>
      <td>737550</td>
      <td>Fritsch, Russel and Anderson</td>
      <td>S1-27722</td>
      <td>15</td>
      <td>70.23</td>
      <td>1053.45</td>
      <td>2014-02-16 18:24:42</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>78</th>
      <td>737550</td>
      <td>Fritsch, Russel and Anderson</td>
      <td>S2-34077</td>
      <td>26</td>
      <td>93.35</td>
      <td>2427.10</td>
      <td>2014-02-20 18:45:43</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>80</th>
      <td>737550</td>
      <td>Fritsch, Russel and Anderson</td>
      <td>S1-93683</td>
      <td>31</td>
      <td>10.52</td>
      <td>326.12</td>
      <td>2014-02-21 13:55:45</td>
      <td>bronze</td>
    </tr>
  </tbody>
</table>
</div>



Now we have all of the data along with the status column filled in. We can do our normal data manipulations using the full suite of pandas capability.

# Using Categories

One of the relatively new functions in pandas is support for categorical data. From the pandas, documentation -

"Categoricals are a pandas data type, which correspond to categorical variables in statistics: a variable, which can take on only a limited, and usually fixed, number of possible values (categories; levels in R). Examples are gender, social class, blood types, country affiliations, observation time or ratings via Likert scales."

For our purposes, the status field is a good candidate for a category type.

You must make sure you have a recent version of pandas installed for this example to work.


```python
pd.__version__
```




    '0.22.0'



First, we typecast it to a category using astype.


```python
all_data_st["status"] = all_data_st["status"].astype("category")
```

This doesn't immediately appear to change anything yet.


```python
all_data_st.head()
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
      <th>account number</th>
      <th>name</th>
      <th>sku</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>383080</td>
      <td>Will LLC</td>
      <td>B1-20000</td>
      <td>7</td>
      <td>33.69</td>
      <td>235.83</td>
      <td>2014-02-01 09:04:59</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>1</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>S1-27722</td>
      <td>11</td>
      <td>21.12</td>
      <td>232.32</td>
      <td>2014-02-01 11:51:46</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>2</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>B1-86481</td>
      <td>3</td>
      <td>35.99</td>
      <td>107.97</td>
      <td>2014-02-01 17:24:32</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>3</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>B1-20000</td>
      <td>23</td>
      <td>78.90</td>
      <td>1814.70</td>
      <td>2014-02-01 19:56:48</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>4</th>
      <td>672390</td>
      <td>Kuhn-Gusikowski</td>
      <td>S1-06532</td>
      <td>48</td>
      <td>55.82</td>
      <td>2679.36</td>
      <td>2014-02-02 03:45:20</td>
      <td>silver</td>
    </tr>
  </tbody>
</table>
</div>



Buy you can see that it is a new data type.


```python
all_data_st.dtypes
```




    account number             int64
    name                      object
    sku                       object
    quantity                   int64
    unit price               float64
    ext price                float64
    date              datetime64[ns]
    status                  category
    dtype: object



Categories get more interesting when you assign order to the categories. Right now, if we call sort on the column, it will sort alphabetically. 


```python
all_data_st.sort_values(by=["status"]).head()
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
      <th>account number</th>
      <th>name</th>
      <th>sku</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>383080</td>
      <td>Will LLC</td>
      <td>B1-20000</td>
      <td>7</td>
      <td>33.69</td>
      <td>235.83</td>
      <td>2014-02-01 09:04:59</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>196</th>
      <td>218895</td>
      <td>Kulas Inc</td>
      <td>S2-83881</td>
      <td>41</td>
      <td>78.27</td>
      <td>3209.07</td>
      <td>2014-01-20 09:37:58</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>197</th>
      <td>383080</td>
      <td>Will LLC</td>
      <td>B1-33364</td>
      <td>26</td>
      <td>90.19</td>
      <td>2344.94</td>
      <td>2014-01-20 09:39:59</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>198</th>
      <td>604255</td>
      <td>Halvorson, Crona and Champlin</td>
      <td>S2-11481</td>
      <td>37</td>
      <td>96.71</td>
      <td>3578.27</td>
      <td>2014-01-20 13:07:28</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>200</th>
      <td>527099</td>
      <td>Sanford and Sons</td>
      <td>B1-05914</td>
      <td>18</td>
      <td>64.32</td>
      <td>1157.76</td>
      <td>2014-01-20 21:40:58</td>
      <td>bronze</td>
    </tr>
  </tbody>
</table>
</div>



We use set_categories to tell it the order we want to use for this category object. In this case, we use the Olympic medal ordering.


```python
 all_data_st["status"].cat.set_categories([ "gold","silver","bronze"],inplace=True)
```

Now, we can sort it so that gold shows on top.


```python
all_data_st.sort_values(by=["status"]).head()
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
      <th>account number</th>
      <th>name</th>
      <th>sku</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>68</th>
      <td>740150</td>
      <td>Barton LLC</td>
      <td>B1-38851</td>
      <td>17</td>
      <td>81.22</td>
      <td>1380.74</td>
      <td>2014-02-17 17:12:16</td>
      <td>gold</td>
    </tr>
    <tr>
      <th>63</th>
      <td>257198</td>
      <td>Cronin, Oberbrunner and Spencer</td>
      <td>S1-27722</td>
      <td>28</td>
      <td>10.21</td>
      <td>285.88</td>
      <td>2014-02-15 17:27:44</td>
      <td>gold</td>
    </tr>
    <tr>
      <th>207</th>
      <td>740150</td>
      <td>Barton LLC</td>
      <td>B1-86481</td>
      <td>20</td>
      <td>30.41</td>
      <td>608.20</td>
      <td>2014-01-22 16:33:51</td>
      <td>gold</td>
    </tr>
    <tr>
      <th>61</th>
      <td>740150</td>
      <td>Barton LLC</td>
      <td>B1-20000</td>
      <td>28</td>
      <td>81.39</td>
      <td>2278.92</td>
      <td>2014-02-15 07:45:16</td>
      <td>gold</td>
    </tr>
    <tr>
      <th>60</th>
      <td>239344</td>
      <td>Stokes LLC</td>
      <td>S2-83881</td>
      <td>30</td>
      <td>43.00</td>
      <td>1290.00</td>
      <td>2014-02-15 02:13:23</td>
      <td>gold</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_data_st["status"].describe()
```




    count        384
    unique         3
    top       bronze
    freq         172
    Name: status, dtype: object



For instance, if you want to take a quick look at how your top tier customers are performaing compared to the bottom. Use groupby to give us the average of the values.


```python
all_data_st.groupby(["status"])["quantity","unit price","ext price"].mean()
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
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
    </tr>
    <tr>
      <th>status</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gold</th>
      <td>24.375000</td>
      <td>53.723889</td>
      <td>1351.944583</td>
    </tr>
    <tr>
      <th>silver</th>
      <td>22.842857</td>
      <td>57.272714</td>
      <td>1320.032214</td>
    </tr>
    <tr>
      <th>bronze</th>
      <td>25.616279</td>
      <td>57.371163</td>
      <td>1472.965930</td>
    </tr>
  </tbody>
</table>
</div>



Of course, you can run multiple aggregation functions on the data to get really useful information 


```python
all_data_st.groupby(["status"])["quantity","unit price","ext price"].agg([np.sum,np.mean, np.std])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">quantity</th>
      <th colspan="3" halign="left">unit price</th>
      <th colspan="3" halign="left">ext price</th>
    </tr>
    <tr>
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>std</th>
      <th>sum</th>
      <th>mean</th>
      <th>std</th>
      <th>sum</th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>status</th>
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
      <th>gold</th>
      <td>1755</td>
      <td>24.375000</td>
      <td>14.575145</td>
      <td>3868.12</td>
      <td>53.723889</td>
      <td>28.740080</td>
      <td>97340.01</td>
      <td>1351.944583</td>
      <td>1182.657312</td>
    </tr>
    <tr>
      <th>silver</th>
      <td>3198</td>
      <td>22.842857</td>
      <td>14.512843</td>
      <td>8018.18</td>
      <td>57.272714</td>
      <td>26.556242</td>
      <td>184804.51</td>
      <td>1320.032214</td>
      <td>1086.384051</td>
    </tr>
    <tr>
      <th>bronze</th>
      <td>4406</td>
      <td>25.616279</td>
      <td>14.136071</td>
      <td>9867.84</td>
      <td>57.371163</td>
      <td>26.857370</td>
      <td>253350.14</td>
      <td>1472.965930</td>
      <td>1116.683843</td>
    </tr>
  </tbody>
</table>
</div>



So, what does this tell you? Well, the data is completely random but my first observation is that we sell more units to our bronze customers than gold. Even when you look at the total dollar value associated with bronze vs. gold, it looks backwards.

Maybe we should look at how many bronze customers we have and see what is going on.

What I plan to do is filter out the unique accounts and see how many gold, silver and bronze customers there are.

I'm purposely stringing a lot of commands together which is not necessarily best practice but does show how powerful pandas can be. Feel free to review my previous articles and play with this command yourself to understand what all these commands mean.


```python
all_data_st.drop_duplicates(subset=["account number","name"]).iloc[:,[0,1,7]].groupby(["status"])["name"].count()
```




    status
    gold      4
    silver    7
    bronze    9
    Name: name, dtype: int64



Ok. This makes a little more sense. We see that we have 9 bronze customers and only 4 customers. That is probably why the volumes are so skewed towards our bronze customers.


```python
all_data_st.head(4)
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
      <th>account number</th>
      <th>name</th>
      <th>sku</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>383080</td>
      <td>Will LLC</td>
      <td>B1-20000</td>
      <td>7</td>
      <td>33.69</td>
      <td>235.83</td>
      <td>2014-02-01 09:04:59</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>1</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>S1-27722</td>
      <td>11</td>
      <td>21.12</td>
      <td>232.32</td>
      <td>2014-02-01 11:51:46</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>2</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>B1-86481</td>
      <td>3</td>
      <td>35.99</td>
      <td>107.97</td>
      <td>2014-02-01 17:24:32</td>
      <td>bronze</td>
    </tr>
    <tr>
      <th>3</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>B1-20000</td>
      <td>23</td>
      <td>78.90</td>
      <td>1814.70</td>
      <td>2014-02-01 19:56:48</td>
      <td>bronze</td>
    </tr>
  </tbody>
</table>
</div>



We will start over and import a clean file


```python
df = pd.read_excel("data/sample-sales-reps.xlsx")

## Set default commision of 3%
df["commission"] = .03
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
      <th>account number</th>
      <th>customer name</th>
      <th>sales rep</th>
      <th>sku</th>
      <th>category</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
      <th>commission</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>GP-14407</td>
      <td>Belt</td>
      <td>19</td>
      <td>88.49</td>
      <td>1681.31</td>
      <td>2015-11-17 05:58:34</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>1</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>FI-01804</td>
      <td>Shirt</td>
      <td>3</td>
      <td>78.07</td>
      <td>234.21</td>
      <td>2016-02-13 04:04:11</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>2</th>
      <td>530925</td>
      <td>Purdy and Sons</td>
      <td>Teagan O'Keefe</td>
      <td>EO-54210</td>
      <td>Shirt</td>
      <td>19</td>
      <td>30.21</td>
      <td>573.99</td>
      <td>2015-08-11 12:44:38</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14406</td>
      <td>Harber, Lubowitz and Fahey</td>
      <td>Esequiel Schinner</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>12</td>
      <td>90.29</td>
      <td>1083.48</td>
      <td>2016-01-23 02:15:50</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>398620</td>
      <td>Brekke Ltd</td>
      <td>Esequiel Schinner</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>5</td>
      <td>72.64</td>
      <td>363.20</td>
      <td>2015-08-10 07:16:03</td>
      <td>0.03</td>
    </tr>
  </tbody>
</table>
</div>



Since shirts are high margin, adjust all products in the shirt categort with a commission rate of 5%


```python
df.loc[df["category"] == "Shirt", ["commission"]] = .05
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
      <th>account number</th>
      <th>customer name</th>
      <th>sales rep</th>
      <th>sku</th>
      <th>category</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
      <th>commission</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>GP-14407</td>
      <td>Belt</td>
      <td>19</td>
      <td>88.49</td>
      <td>1681.31</td>
      <td>2015-11-17 05:58:34</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>1</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>FI-01804</td>
      <td>Shirt</td>
      <td>3</td>
      <td>78.07</td>
      <td>234.21</td>
      <td>2016-02-13 04:04:11</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>530925</td>
      <td>Purdy and Sons</td>
      <td>Teagan O'Keefe</td>
      <td>EO-54210</td>
      <td>Shirt</td>
      <td>19</td>
      <td>30.21</td>
      <td>573.99</td>
      <td>2015-08-11 12:44:38</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14406</td>
      <td>Harber, Lubowitz and Fahey</td>
      <td>Esequiel Schinner</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>12</td>
      <td>90.29</td>
      <td>1083.48</td>
      <td>2016-01-23 02:15:50</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>398620</td>
      <td>Brekke Ltd</td>
      <td>Esequiel Schinner</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>5</td>
      <td>72.64</td>
      <td>363.20</td>
      <td>2015-08-10 07:16:03</td>
      <td>0.05</td>
    </tr>
  </tbody>
</table>
</div>



Since there is a special program for selling 10 or more belts in a transaction, you get 7% commission!


```python
df.loc[(df["category"] == "Belt") & (df["quantity"] >= 10), ["commission"]] = .04
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
      <th>account number</th>
      <th>customer name</th>
      <th>sales rep</th>
      <th>sku</th>
      <th>category</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
      <th>commission</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>GP-14407</td>
      <td>Belt</td>
      <td>19</td>
      <td>88.49</td>
      <td>1681.31</td>
      <td>2015-11-17 05:58:34</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>FI-01804</td>
      <td>Shirt</td>
      <td>3</td>
      <td>78.07</td>
      <td>234.21</td>
      <td>2016-02-13 04:04:11</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>530925</td>
      <td>Purdy and Sons</td>
      <td>Teagan O'Keefe</td>
      <td>EO-54210</td>
      <td>Shirt</td>
      <td>19</td>
      <td>30.21</td>
      <td>573.99</td>
      <td>2015-08-11 12:44:38</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14406</td>
      <td>Harber, Lubowitz and Fahey</td>
      <td>Esequiel Schinner</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>12</td>
      <td>90.29</td>
      <td>1083.48</td>
      <td>2016-01-23 02:15:50</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>398620</td>
      <td>Brekke Ltd</td>
      <td>Esequiel Schinner</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>5</td>
      <td>72.64</td>
      <td>363.20</td>
      <td>2015-08-10 07:16:03</td>
      <td>0.05</td>
    </tr>
  </tbody>
</table>
</div>





Finally, some transactions can get a bonus and a commission increase.



```python
df["bonus"] = 0
df.loc[(df["category"] == "Shoes") & (df["ext price"] >= 1000 ), ["bonus", "commission"]] = 250, 0.045

```

Calculate the compensation at the line item level


```python
df["comp"] = df["commission"] * df["ext price"] + df["bonus"]
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
      <th>account number</th>
      <th>customer name</th>
      <th>sales rep</th>
      <th>sku</th>
      <th>category</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
      <th>commission</th>
      <th>bonus</th>
      <th>comp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>GP-14407</td>
      <td>Belt</td>
      <td>19</td>
      <td>88.49</td>
      <td>1681.31</td>
      <td>2015-11-17 05:58:34</td>
      <td>0.04</td>
      <td>0</td>
      <td>67.2524</td>
    </tr>
    <tr>
      <th>1</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>FI-01804</td>
      <td>Shirt</td>
      <td>3</td>
      <td>78.07</td>
      <td>234.21</td>
      <td>2016-02-13 04:04:11</td>
      <td>0.05</td>
      <td>0</td>
      <td>11.7105</td>
    </tr>
    <tr>
      <th>2</th>
      <td>530925</td>
      <td>Purdy and Sons</td>
      <td>Teagan O'Keefe</td>
      <td>EO-54210</td>
      <td>Shirt</td>
      <td>19</td>
      <td>30.21</td>
      <td>573.99</td>
      <td>2015-08-11 12:44:38</td>
      <td>0.05</td>
      <td>0</td>
      <td>28.6995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14406</td>
      <td>Harber, Lubowitz and Fahey</td>
      <td>Esequiel Schinner</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>12</td>
      <td>90.29</td>
      <td>1083.48</td>
      <td>2016-01-23 02:15:50</td>
      <td>0.05</td>
      <td>0</td>
      <td>54.1740</td>
    </tr>
    <tr>
      <th>4</th>
      <td>398620</td>
      <td>Brekke Ltd</td>
      <td>Esequiel Schinner</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>5</td>
      <td>72.64</td>
      <td>363.20</td>
      <td>2015-08-10 07:16:03</td>
      <td>0.05</td>
      <td>0</td>
      <td>18.1600</td>
    </tr>
  </tbody>
</table>
</div>



Calculate the commissions by sales rep


```python
df.groupby(["sales rep"])["comp"].sum().round(2)


```




    sales rep
    Ansley Cummings       2699.69
    Beth Skiles           3664.16
    Esequiel Schinner    12841.28
    Loring Predovic      13115.42
    Shannen Hudson        6541.78
    Teagan O'Keefe       10931.30
    Trish Deckow          7641.91
    Name: comp, dtype: float64




```python
df["date"] = pd.to_datetime(df['date'])

df["month"] = df["date"].dt.month
```

Now what about on a monthly basis


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
      <th>account number</th>
      <th>customer name</th>
      <th>sales rep</th>
      <th>sku</th>
      <th>category</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
      <th>commission</th>
      <th>bonus</th>
      <th>comp</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>GP-14407</td>
      <td>Belt</td>
      <td>19</td>
      <td>88.49</td>
      <td>1681.31</td>
      <td>2015-11-17 05:58:34</td>
      <td>0.04</td>
      <td>0</td>
      <td>67.2524</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>FI-01804</td>
      <td>Shirt</td>
      <td>3</td>
      <td>78.07</td>
      <td>234.21</td>
      <td>2016-02-13 04:04:11</td>
      <td>0.05</td>
      <td>0</td>
      <td>11.7105</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>530925</td>
      <td>Purdy and Sons</td>
      <td>Teagan O'Keefe</td>
      <td>EO-54210</td>
      <td>Shirt</td>
      <td>19</td>
      <td>30.21</td>
      <td>573.99</td>
      <td>2015-08-11 12:44:38</td>
      <td>0.05</td>
      <td>0</td>
      <td>28.6995</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14406</td>
      <td>Harber, Lubowitz and Fahey</td>
      <td>Esequiel Schinner</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>12</td>
      <td>90.29</td>
      <td>1083.48</td>
      <td>2016-01-23 02:15:50</td>
      <td>0.05</td>
      <td>0</td>
      <td>54.1740</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>398620</td>
      <td>Brekke Ltd</td>
      <td>Esequiel Schinner</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>5</td>
      <td>72.64</td>
      <td>363.20</td>
      <td>2015-08-10 07:16:03</td>
      <td>0.05</td>
      <td>0</td>
      <td>18.1600</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby(["month","sales rep"])["comp"].sum().round(2)
```




    month  sales rep        
    1      Ansley Cummings       157.10
           Beth Skiles            78.09
           Esequiel Schinner    1481.78
           Loring Predovic       655.96
           Shannen Hudson        319.13
           Teagan O'Keefe        732.10
           Trish Deckow          305.58
    2      Ansley Cummings       347.06
           Beth Skiles          1262.36
           Esequiel Schinner     741.87
           Loring Predovic      1794.22
           Shannen Hudson        524.58
           Teagan O'Keefe        893.54
           Trish Deckow          430.81
    3      Ansley Cummings       362.13
           Beth Skiles           439.53
           Esequiel Schinner    1323.87
           Loring Predovic      1183.59
           Shannen Hudson        474.90
           Teagan O'Keefe       1064.76
           Trish Deckow          796.12
    4      Ansley Cummings       123.27
           Beth Skiles           121.12
           Esequiel Schinner    1478.78
           Loring Predovic       907.41
           Shannen Hudson        514.77
           Teagan O'Keefe        593.64
           Trish Deckow          347.04
    5      Ansley Cummings       101.29
           Beth Skiles           347.73
                                 ...   
    8      Teagan O'Keefe        477.33
           Trish Deckow         1493.39
    9      Ansley Cummings       109.70
           Beth Skiles           145.43
           Esequiel Schinner     178.20
           Loring Predovic       978.46
           Shannen Hudson        765.28
           Teagan O'Keefe       1086.80
           Trish Deckow          435.42
    10     Ansley Cummings       510.49
           Beth Skiles           204.40
           Esequiel Schinner     994.76
           Loring Predovic      1954.61
           Shannen Hudson        990.48
           Teagan O'Keefe       1111.47
           Trish Deckow          625.12
    11     Ansley Cummings        80.41
           Beth Skiles           464.13
           Esequiel Schinner     928.05
           Loring Predovic      1253.52
           Shannen Hudson        567.31
           Teagan O'Keefe        554.03
           Trish Deckow          281.69
    12     Ansley Cummings       288.82
           Beth Skiles           131.40
           Esequiel Schinner    1235.69
           Loring Predovic       660.44
           Shannen Hudson        421.24
           Teagan O'Keefe        421.16
           Trish Deckow          848.57
    Name: comp, Length: 84, dtype: float64



You can do this for the entire numerical dataframe, withou creating a month variable like follows


```python
df.set_index('date').groupby('sales rep').resample("M").sum().head(20)
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
      <th>account number</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>commission</th>
      <th>bonus</th>
      <th>comp</th>
      <th>month</th>
    </tr>
    <tr>
      <th>sales rep</th>
      <th>date</th>
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
      <th rowspan="13" valign="top">Ansley Cummings</th>
      <th>2015-06-30</th>
      <td>3593984</td>
      <td>59</td>
      <td>196.43</td>
      <td>2214.13</td>
      <td>0.165</td>
      <td>250</td>
      <td>342.04250</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>3593984</td>
      <td>43</td>
      <td>152.77</td>
      <td>1460.69</td>
      <td>0.160</td>
      <td>0</td>
      <td>58.43250</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2015-08-31</th>
      <td>7187968</td>
      <td>77</td>
      <td>437.26</td>
      <td>4080.37</td>
      <td>0.340</td>
      <td>0</td>
      <td>171.32280</td>
      <td>64</td>
    </tr>
    <tr>
      <th>2015-09-30</th>
      <td>6289472</td>
      <td>64</td>
      <td>398.20</td>
      <td>2691.38</td>
      <td>0.280</td>
      <td>0</td>
      <td>109.69920</td>
      <td>63</td>
    </tr>
    <tr>
      <th>2015-10-31</th>
      <td>6289472</td>
      <td>86</td>
      <td>437.41</td>
      <td>5803.84</td>
      <td>0.315</td>
      <td>250</td>
      <td>510.48870</td>
      <td>70</td>
    </tr>
    <tr>
      <th>2015-11-30</th>
      <td>3593984</td>
      <td>25</td>
      <td>259.40</td>
      <td>1715.97</td>
      <td>0.160</td>
      <td>0</td>
      <td>80.40890</td>
      <td>44</td>
    </tr>
    <tr>
      <th>2015-12-31</th>
      <td>9883456</td>
      <td>139</td>
      <td>465.39</td>
      <td>6820.11</td>
      <td>0.450</td>
      <td>0</td>
      <td>288.81630</td>
      <td>132</td>
    </tr>
    <tr>
      <th>2016-01-31</th>
      <td>7187968</td>
      <td>85</td>
      <td>413.52</td>
      <td>4346.59</td>
      <td>0.310</td>
      <td>0</td>
      <td>157.10410</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2016-02-29</th>
      <td>4492480</td>
      <td>56</td>
      <td>233.69</td>
      <td>2561.57</td>
      <td>0.185</td>
      <td>250</td>
      <td>347.06005</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2016-03-31</th>
      <td>3593984</td>
      <td>43</td>
      <td>260.11</td>
      <td>2553.24</td>
      <td>0.175</td>
      <td>250</td>
      <td>362.13085</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2016-04-30</th>
      <td>4492480</td>
      <td>55</td>
      <td>276.44</td>
      <td>2855.68</td>
      <td>0.200</td>
      <td>0</td>
      <td>123.26960</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2016-05-31</th>
      <td>3593984</td>
      <td>36</td>
      <td>282.24</td>
      <td>2151.62</td>
      <td>0.190</td>
      <td>0</td>
      <td>101.28800</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2016-06-30</th>
      <td>3593984</td>
      <td>15</td>
      <td>257.26</td>
      <td>980.25</td>
      <td>0.180</td>
      <td>0</td>
      <td>47.62630</td>
      <td>24</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Beth Skiles</th>
      <th>2015-06-30</th>
      <td>846366</td>
      <td>37</td>
      <td>161.44</td>
      <td>1763.43</td>
      <td>0.150</td>
      <td>0</td>
      <td>88.17150</td>
      <td>18</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>1692732</td>
      <td>64</td>
      <td>235.55</td>
      <td>2468.03</td>
      <td>0.190</td>
      <td>0</td>
      <td>81.93400</td>
      <td>42</td>
    </tr>
    <tr>
      <th>2015-08-31</th>
      <td>3385464</td>
      <td>123</td>
      <td>699.43</td>
      <td>6275.63</td>
      <td>0.470</td>
      <td>0</td>
      <td>283.95210</td>
      <td>96</td>
    </tr>
    <tr>
      <th>2015-09-30</th>
      <td>1692732</td>
      <td>68</td>
      <td>326.76</td>
      <td>3374.01</td>
      <td>0.240</td>
      <td>0</td>
      <td>145.42650</td>
      <td>54</td>
    </tr>
    <tr>
      <th>2015-10-31</th>
      <td>3103342</td>
      <td>106</td>
      <td>591.17</td>
      <td>5248.15</td>
      <td>0.410</td>
      <td>0</td>
      <td>204.40310</td>
      <td>110</td>
    </tr>
    <tr>
      <th>2015-11-30</th>
      <td>1974854</td>
      <td>78</td>
      <td>412.76</td>
      <td>4496.55</td>
      <td>0.315</td>
      <td>250</td>
      <td>464.13370</td>
      <td>77</td>
    </tr>
    <tr>
      <th>2015-12-31</th>
      <td>2539098</td>
      <td>55</td>
      <td>480.51</td>
      <td>2970.65</td>
      <td>0.360</td>
      <td>0</td>
      <td>131.39730</td>
      <td>108</td>
    </tr>
  </tbody>
</table>
</div>



What if you are only interested in Mondays


```python
df.set_index('date').groupby('sales rep').resample("W-Mon").sum().head(20)
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
      <th>account number</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>commission</th>
      <th>bonus</th>
      <th>comp</th>
      <th>month</th>
    </tr>
    <tr>
      <th>sales rep</th>
      <th>date</th>
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
      <th rowspan="20" valign="top">Ansley Cummings</th>
      <th>2015-06-15</th>
      <td>1796992</td>
      <td>39</td>
      <td>72.79</td>
      <td>1436.45</td>
      <td>0.075</td>
      <td>250</td>
      <td>309.1255</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2015-06-22</th>
      <td>898496</td>
      <td>18</td>
      <td>33.15</td>
      <td>596.70</td>
      <td>0.040</td>
      <td>0</td>
      <td>23.8680</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2015-06-29</th>
      <td>898496</td>
      <td>2</td>
      <td>90.49</td>
      <td>180.98</td>
      <td>0.050</td>
      <td>0</td>
      <td>9.0490</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2015-07-06</th>
      <td>898496</td>
      <td>11</td>
      <td>22.99</td>
      <td>252.89</td>
      <td>0.050</td>
      <td>0</td>
      <td>12.6445</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2015-07-13</th>
      <td>898496</td>
      <td>7</td>
      <td>83.34</td>
      <td>583.38</td>
      <td>0.030</td>
      <td>0</td>
      <td>17.5014</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2015-07-20</th>
      <td>898496</td>
      <td>17</td>
      <td>28.10</td>
      <td>477.70</td>
      <td>0.050</td>
      <td>0</td>
      <td>23.8850</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2015-07-27</th>
      <td>898496</td>
      <td>8</td>
      <td>18.34</td>
      <td>146.72</td>
      <td>0.030</td>
      <td>0</td>
      <td>4.4016</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2015-08-03</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0</td>
      <td>0.0000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-08-10</th>
      <td>2695488</td>
      <td>36</td>
      <td>112.58</td>
      <td>1191.71</td>
      <td>0.120</td>
      <td>0</td>
      <td>50.4873</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2015-08-17</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0</td>
      <td>0.0000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-08-24</th>
      <td>2695488</td>
      <td>30</td>
      <td>170.84</td>
      <td>1964.79</td>
      <td>0.140</td>
      <td>0</td>
      <td>89.4372</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2015-08-31</th>
      <td>1796992</td>
      <td>11</td>
      <td>153.84</td>
      <td>923.87</td>
      <td>0.080</td>
      <td>0</td>
      <td>31.3983</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2015-09-07</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0</td>
      <td>0.0000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-09-14</th>
      <td>2695488</td>
      <td>47</td>
      <td>89.22</td>
      <td>1432.40</td>
      <td>0.120</td>
      <td>0</td>
      <td>59.1218</td>
      <td>27</td>
    </tr>
    <tr>
      <th>2015-09-21</th>
      <td>1796992</td>
      <td>8</td>
      <td>162.20</td>
      <td>640.40</td>
      <td>0.100</td>
      <td>0</td>
      <td>32.0200</td>
      <td>18</td>
    </tr>
    <tr>
      <th>2015-09-28</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0</td>
      <td>0.0000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-10-05</th>
      <td>1796992</td>
      <td>9</td>
      <td>146.78</td>
      <td>618.58</td>
      <td>0.060</td>
      <td>0</td>
      <td>18.5574</td>
      <td>18</td>
    </tr>
    <tr>
      <th>2015-10-12</th>
      <td>3593984</td>
      <td>42</td>
      <td>193.83</td>
      <td>2216.34</td>
      <td>0.170</td>
      <td>0</td>
      <td>87.6666</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2015-10-19</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0</td>
      <td>0.0000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-10-26</th>
      <td>2695488</td>
      <td>44</td>
      <td>243.58</td>
      <td>3587.50</td>
      <td>0.145</td>
      <td>250</td>
      <td>422.8221</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>


for a specific month use:
df.groupby(['name', 'sku', pd.Grouper(key='date', freq='A-DEC')])['ext price'].sum()


Here is an aggregation function that is sometimes usefull


```python
df.agg({'ext price': ['sum', 'mean'], 'quantity': ['sum', 'mean'], 'unit price': ['mean']})
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
      <th>ext price</th>
      <th>quantity</th>
      <th>unit price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>571.75346</td>
      <td>10.411333</td>
      <td>55.316493</td>
    </tr>
    <tr>
      <th>sum</th>
      <td>857630.19000</td>
      <td>15617.000000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



You can create custom functions


```python
get_max = lambda x: x.value_counts(dropna=False).index[0]
get_max.__name__ = "most frequent" # required for row label

df.agg({'ext price': ['sum', 'mean'], 'quantity': ['sum', 'mean'], 'unit price': ['mean'], 'sku': [get_max]})
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
      <th>ext price</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>sku</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>571.75346</td>
      <td>10.411333</td>
      <td>55.316493</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>most frequent</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>TL-23025</td>
    </tr>
    <tr>
      <th>sum</th>
      <td>857630.19000</td>
      <td>15617.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



It is nice to have an ordered dictionary


```python
import collections
f = collections.OrderedDict([('ext price', ['sum', 'mean']), ('quantity', ['sum', 'mean']), ('sku', [get_max])])
df.agg(f)
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
      <th>ext price</th>
      <th>quantity</th>
      <th>sku</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>571.75346</td>
      <td>10.411333</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>most frequent</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>TL-23025</td>
    </tr>
    <tr>
      <th>sum</th>
      <td>857630.19000</td>
      <td>15617.000000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



You can of course use any of the extensions available like below to further analysis and filtering. 


```python
import qgrid
from IPython.display import display

qgrid_widget = qgrid.show_grid(df, show_toolbar=True)
```


```python
## Not sure why this is not working - giving it a skip for now
```


```python
qgrid_widget
```


    QgridWidget(grid_options={'fullWidthRows': True, 'syncColumnCellResize': True, 'forceFitColumns': True, 'defau…



```python
qgrid_widget.get_changed_df()
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
      <th>account number</th>
      <th>customer name</th>
      <th>sales rep</th>
      <th>sku</th>
      <th>category</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
      <th>commission</th>
      <th>bonus</th>
      <th>comp</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>GP-14407</td>
      <td>Belt</td>
      <td>19</td>
      <td>88.49</td>
      <td>1681.31</td>
      <td>2015-11-17 05:58:34</td>
      <td>0.040</td>
      <td>0</td>
      <td>67.2524</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>FI-01804</td>
      <td>Shirt</td>
      <td>3</td>
      <td>78.07</td>
      <td>234.21</td>
      <td>2016-02-13 04:04:11</td>
      <td>0.050</td>
      <td>0</td>
      <td>11.7105</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>530925</td>
      <td>Purdy and Sons</td>
      <td>Teagan O'Keefe</td>
      <td>EO-54210</td>
      <td>Shirt</td>
      <td>19</td>
      <td>30.21</td>
      <td>573.99</td>
      <td>2015-08-11 12:44:38</td>
      <td>0.050</td>
      <td>0</td>
      <td>28.6995</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14406</td>
      <td>Harber, Lubowitz and Fahey</td>
      <td>Esequiel Schinner</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>12</td>
      <td>90.29</td>
      <td>1083.48</td>
      <td>2016-01-23 02:15:50</td>
      <td>0.050</td>
      <td>0</td>
      <td>54.1740</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>398620</td>
      <td>Brekke Ltd</td>
      <td>Esequiel Schinner</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>5</td>
      <td>72.64</td>
      <td>363.20</td>
      <td>2015-08-10 07:16:03</td>
      <td>0.050</td>
      <td>0</td>
      <td>18.1600</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>282122</td>
      <td>Connelly, Abshire and Von</td>
      <td>Beth Skiles</td>
      <td>GJ-90272</td>
      <td>Shoes</td>
      <td>20</td>
      <td>96.62</td>
      <td>1932.40</td>
      <td>2016-03-17 10:19:05</td>
      <td>0.045</td>
      <td>250</td>
      <td>336.9580</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>398620</td>
      <td>Brekke Ltd</td>
      <td>Esequiel Schinner</td>
      <td>DU-87462</td>
      <td>Shirt</td>
      <td>10</td>
      <td>67.64</td>
      <td>676.40</td>
      <td>2015-11-25 22:05:36</td>
      <td>0.050</td>
      <td>0</td>
      <td>33.8200</td>
      <td>11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>218667</td>
      <td>Jaskolski-O'Hara</td>
      <td>Trish Deckow</td>
      <td>DU-87462</td>
      <td>Shirt</td>
      <td>11</td>
      <td>91.86</td>
      <td>1010.46</td>
      <td>2016-04-24 15:05:58</td>
      <td>0.050</td>
      <td>0</td>
      <td>50.5230</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>398620</td>
      <td>Brekke Ltd</td>
      <td>Esequiel Schinner</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>18</td>
      <td>98.67</td>
      <td>1776.06</td>
      <td>2015-08-06 08:09:56</td>
      <td>0.050</td>
      <td>0</td>
      <td>88.8030</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>14406</td>
      <td>Harber, Lubowitz and Fahey</td>
      <td>Esequiel Schinner</td>
      <td>GP-14407</td>
      <td>Belt</td>
      <td>12</td>
      <td>64.48</td>
      <td>773.76</td>
      <td>2016-01-08 09:52:04</td>
      <td>0.040</td>
      <td>0</td>
      <td>30.9504</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>887145</td>
      <td>Gislason LLC</td>
      <td>Loring Predovic</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>20</td>
      <td>92.87</td>
      <td>1857.40</td>
      <td>2016-05-22 06:09:58</td>
      <td>0.050</td>
      <td>0</td>
      <td>92.8700</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>93583</td>
      <td>Hegmann-Howell</td>
      <td>Esequiel Schinner</td>
      <td>HZ-54995</td>
      <td>Belt</td>
      <td>17</td>
      <td>40.56</td>
      <td>689.52</td>
      <td>2015-07-05 01:05:52</td>
      <td>0.040</td>
      <td>0</td>
      <td>27.5808</td>
      <td>7</td>
    </tr>
    <tr>
      <th>12</th>
      <td>280749</td>
      <td>Douglas PLC</td>
      <td>Teagan O'Keefe</td>
      <td>TK-29646</td>
      <td>Shoes</td>
      <td>17</td>
      <td>42.61</td>
      <td>724.37</td>
      <td>2016-05-16 19:11:55</td>
      <td>0.030</td>
      <td>0</td>
      <td>21.7311</td>
      <td>5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>453258</td>
      <td>Runolfsson-Bayer</td>
      <td>Shannen Hudson</td>
      <td>FI-01804</td>
      <td>Shirt</td>
      <td>20</td>
      <td>12.31</td>
      <td>246.20</td>
      <td>2015-08-24 21:32:26</td>
      <td>0.050</td>
      <td>0</td>
      <td>12.3100</td>
      <td>8</td>
    </tr>
    <tr>
      <th>14</th>
      <td>453258</td>
      <td>Runolfsson-Bayer</td>
      <td>Shannen Hudson</td>
      <td>EO-54210</td>
      <td>Shirt</td>
      <td>20</td>
      <td>67.95</td>
      <td>1359.00</td>
      <td>2015-11-24 15:04:30</td>
      <td>0.050</td>
      <td>0</td>
      <td>67.9500</td>
      <td>11</td>
    </tr>
    <tr>
      <th>15</th>
      <td>280749</td>
      <td>Douglas PLC</td>
      <td>Teagan O'Keefe</td>
      <td>GJ-90272</td>
      <td>Shoes</td>
      <td>17</td>
      <td>23.20</td>
      <td>394.40</td>
      <td>2015-09-12 13:16:56</td>
      <td>0.030</td>
      <td>0</td>
      <td>11.8320</td>
      <td>9</td>
    </tr>
    <tr>
      <th>16</th>
      <td>453258</td>
      <td>Runolfsson-Bayer</td>
      <td>Shannen Hudson</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>2</td>
      <td>30.23</td>
      <td>60.46</td>
      <td>2016-03-21 11:31:24</td>
      <td>0.050</td>
      <td>0</td>
      <td>3.0230</td>
      <td>3</td>
    </tr>
    <tr>
      <th>17</th>
      <td>280749</td>
      <td>Douglas PLC</td>
      <td>Teagan O'Keefe</td>
      <td>GP-14407</td>
      <td>Belt</td>
      <td>3</td>
      <td>22.64</td>
      <td>67.92</td>
      <td>2015-12-02 01:55:28</td>
      <td>0.030</td>
      <td>0</td>
      <td>2.0376</td>
      <td>12</td>
    </tr>
    <tr>
      <th>18</th>
      <td>398620</td>
      <td>Brekke Ltd</td>
      <td>Esequiel Schinner</td>
      <td>HZ-54995</td>
      <td>Belt</td>
      <td>1</td>
      <td>36.20</td>
      <td>36.20</td>
      <td>2015-12-07 18:45:54</td>
      <td>0.030</td>
      <td>0</td>
      <td>1.0860</td>
      <td>12</td>
    </tr>
    <tr>
      <th>19</th>
      <td>251881</td>
      <td>Zulauf-Grady</td>
      <td>Teagan O'Keefe</td>
      <td>DU-87462</td>
      <td>Shirt</td>
      <td>9</td>
      <td>82.06</td>
      <td>738.54</td>
      <td>2016-02-23 17:41:41</td>
      <td>0.050</td>
      <td>0</td>
      <td>36.9270</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>575704</td>
      <td>Lindgren, Thompson and Kirlin</td>
      <td>Teagan O'Keefe</td>
      <td>DU-87462</td>
      <td>Shirt</td>
      <td>5</td>
      <td>28.51</td>
      <td>142.55</td>
      <td>2016-03-01 19:34:33</td>
      <td>0.050</td>
      <td>0</td>
      <td>7.1275</td>
      <td>3</td>
    </tr>
    <tr>
      <th>21</th>
      <td>887145</td>
      <td>Gislason LLC</td>
      <td>Loring Predovic</td>
      <td>ZY-38455</td>
      <td>Shirt</td>
      <td>15</td>
      <td>95.60</td>
      <td>1434.00</td>
      <td>2015-10-08 03:01:22</td>
      <td>0.050</td>
      <td>0</td>
      <td>71.7000</td>
      <td>10</td>
    </tr>
    <tr>
      <th>22</th>
      <td>575704</td>
      <td>Lindgren, Thompson and Kirlin</td>
      <td>Teagan O'Keefe</td>
      <td>TK-29646</td>
      <td>Shoes</td>
      <td>11</td>
      <td>30.80</td>
      <td>338.80</td>
      <td>2015-11-25 06:07:47</td>
      <td>0.030</td>
      <td>0</td>
      <td>10.1640</td>
      <td>11</td>
    </tr>
    <tr>
      <th>23</th>
      <td>282122</td>
      <td>Connelly, Abshire and Von</td>
      <td>Beth Skiles</td>
      <td>HZ-54995</td>
      <td>Belt</td>
      <td>6</td>
      <td>64.98</td>
      <td>389.88</td>
      <td>2016-01-15 21:07:30</td>
      <td>0.030</td>
      <td>0</td>
      <td>11.6964</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>758030</td>
      <td>Kilback-Abernathy</td>
      <td>Trish Deckow</td>
      <td>TK-29646</td>
      <td>Shoes</td>
      <td>19</td>
      <td>26.90</td>
      <td>511.10</td>
      <td>2015-09-28 14:33:34</td>
      <td>0.030</td>
      <td>0</td>
      <td>15.3330</td>
      <td>9</td>
    </tr>
    <tr>
      <th>25</th>
      <td>280749</td>
      <td>Douglas PLC</td>
      <td>Teagan O'Keefe</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>15</td>
      <td>30.34</td>
      <td>455.10</td>
      <td>2016-03-29 21:19:08</td>
      <td>0.050</td>
      <td>0</td>
      <td>22.7550</td>
      <td>3</td>
    </tr>
    <tr>
      <th>26</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>DU-87462</td>
      <td>Shirt</td>
      <td>2</td>
      <td>61.01</td>
      <td>122.02</td>
      <td>2016-05-19 00:05:12</td>
      <td>0.050</td>
      <td>0</td>
      <td>6.1010</td>
      <td>5</td>
    </tr>
    <tr>
      <th>27</th>
      <td>898496</td>
      <td>Weissnat, Veum and Barton</td>
      <td>Ansley Cummings</td>
      <td>TL-23025</td>
      <td>Shoes</td>
      <td>8</td>
      <td>18.34</td>
      <td>146.72</td>
      <td>2015-07-24 17:48:46</td>
      <td>0.030</td>
      <td>0</td>
      <td>4.4016</td>
      <td>7</td>
    </tr>
    <tr>
      <th>28</th>
      <td>530925</td>
      <td>Purdy and Sons</td>
      <td>Teagan O'Keefe</td>
      <td>DU-87462</td>
      <td>Shirt</td>
      <td>10</td>
      <td>18.40</td>
      <td>184.00</td>
      <td>2016-03-05 23:45:57</td>
      <td>0.050</td>
      <td>0</td>
      <td>9.2000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>29</th>
      <td>752312</td>
      <td>Watsica-Pfannerstill</td>
      <td>Loring Predovic</td>
      <td>GP-14407</td>
      <td>Belt</td>
      <td>15</td>
      <td>17.93</td>
      <td>268.95</td>
      <td>2016-03-18 07:17:35</td>
      <td>0.040</td>
      <td>0</td>
      <td>10.7580</td>
      <td>3</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1470</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>HZ-54995</td>
      <td>Belt</td>
      <td>18</td>
      <td>17.93</td>
      <td>322.74</td>
      <td>2016-04-28 01:57:00</td>
      <td>0.040</td>
      <td>0</td>
      <td>12.9096</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1471</th>
      <td>218667</td>
      <td>Jaskolski-O'Hara</td>
      <td>Trish Deckow</td>
      <td>TK-29646</td>
      <td>Shoes</td>
      <td>13</td>
      <td>57.80</td>
      <td>751.40</td>
      <td>2015-11-09 12:17:52</td>
      <td>0.030</td>
      <td>0</td>
      <td>22.5420</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1472</th>
      <td>887145</td>
      <td>Gislason LLC</td>
      <td>Loring Predovic</td>
      <td>GP-14407</td>
      <td>Belt</td>
      <td>5</td>
      <td>51.82</td>
      <td>259.10</td>
      <td>2016-06-08 11:25:05</td>
      <td>0.030</td>
      <td>0</td>
      <td>7.7730</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1473</th>
      <td>93583</td>
      <td>Hegmann-Howell</td>
      <td>Esequiel Schinner</td>
      <td>ZY-38455</td>
      <td>Shirt</td>
      <td>10</td>
      <td>86.25</td>
      <td>862.50</td>
      <td>2016-05-10 23:48:57</td>
      <td>0.050</td>
      <td>0</td>
      <td>43.1250</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1474</th>
      <td>530925</td>
      <td>Purdy and Sons</td>
      <td>Teagan O'Keefe</td>
      <td>FI-01804</td>
      <td>Shirt</td>
      <td>13</td>
      <td>10.13</td>
      <td>131.69</td>
      <td>2016-04-01 20:39:41</td>
      <td>0.050</td>
      <td>0</td>
      <td>6.5845</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1475</th>
      <td>752312</td>
      <td>Watsica-Pfannerstill</td>
      <td>Loring Predovic</td>
      <td>HZ-54995</td>
      <td>Belt</td>
      <td>18</td>
      <td>23.48</td>
      <td>422.64</td>
      <td>2015-07-25 05:51:10</td>
      <td>0.040</td>
      <td>0</td>
      <td>16.9056</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1476</th>
      <td>758030</td>
      <td>Kilback-Abernathy</td>
      <td>Trish Deckow</td>
      <td>TK-29646</td>
      <td>Shoes</td>
      <td>7</td>
      <td>86.67</td>
      <td>606.69</td>
      <td>2015-08-19 15:35:31</td>
      <td>0.030</td>
      <td>0</td>
      <td>18.2007</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1477</th>
      <td>398620</td>
      <td>Brekke Ltd</td>
      <td>Esequiel Schinner</td>
      <td>GP-14407</td>
      <td>Belt</td>
      <td>7</td>
      <td>16.52</td>
      <td>115.64</td>
      <td>2016-04-07 10:25:42</td>
      <td>0.030</td>
      <td>0</td>
      <td>3.4692</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1478</th>
      <td>218667</td>
      <td>Jaskolski-O'Hara</td>
      <td>Trish Deckow</td>
      <td>GJ-90272</td>
      <td>Shoes</td>
      <td>6</td>
      <td>36.44</td>
      <td>218.64</td>
      <td>2015-11-02 20:55:11</td>
      <td>0.030</td>
      <td>0</td>
      <td>6.5592</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1479</th>
      <td>14406</td>
      <td>Harber, Lubowitz and Fahey</td>
      <td>Esequiel Schinner</td>
      <td>HZ-54995</td>
      <td>Belt</td>
      <td>5</td>
      <td>48.52</td>
      <td>242.60</td>
      <td>2016-05-22 12:34:35</td>
      <td>0.030</td>
      <td>0</td>
      <td>7.2780</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1480</th>
      <td>218667</td>
      <td>Jaskolski-O'Hara</td>
      <td>Trish Deckow</td>
      <td>TL-23025</td>
      <td>Shoes</td>
      <td>14</td>
      <td>88.33</td>
      <td>1236.62</td>
      <td>2015-07-12 15:59:56</td>
      <td>0.045</td>
      <td>250</td>
      <td>305.6479</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1481</th>
      <td>530925</td>
      <td>Purdy and Sons</td>
      <td>Teagan O'Keefe</td>
      <td>HZ-54995</td>
      <td>Belt</td>
      <td>9</td>
      <td>62.85</td>
      <td>565.65</td>
      <td>2015-08-12 17:07:20</td>
      <td>0.030</td>
      <td>0</td>
      <td>16.9695</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1482</th>
      <td>575704</td>
      <td>Lindgren, Thompson and Kirlin</td>
      <td>Teagan O'Keefe</td>
      <td>GP-14407</td>
      <td>Belt</td>
      <td>9</td>
      <td>55.57</td>
      <td>500.13</td>
      <td>2016-02-27 03:20:13</td>
      <td>0.030</td>
      <td>0</td>
      <td>15.0039</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1483</th>
      <td>758030</td>
      <td>Kilback-Abernathy</td>
      <td>Trish Deckow</td>
      <td>GJ-90272</td>
      <td>Shoes</td>
      <td>8</td>
      <td>98.87</td>
      <td>790.96</td>
      <td>2016-02-10 16:51:59</td>
      <td>0.030</td>
      <td>0</td>
      <td>23.7288</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1484</th>
      <td>136521</td>
      <td>Labadie and Sons</td>
      <td>Esequiel Schinner</td>
      <td>FI-01804</td>
      <td>Shirt</td>
      <td>15</td>
      <td>99.43</td>
      <td>1491.45</td>
      <td>2015-12-17 21:58:31</td>
      <td>0.050</td>
      <td>0</td>
      <td>74.5725</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1485</th>
      <td>898496</td>
      <td>Weissnat, Veum and Barton</td>
      <td>Ansley Cummings</td>
      <td>GJ-90272</td>
      <td>Shoes</td>
      <td>18</td>
      <td>11.05</td>
      <td>198.90</td>
      <td>2016-04-01 10:19:26</td>
      <td>0.030</td>
      <td>0</td>
      <td>5.9670</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1486</th>
      <td>136521</td>
      <td>Labadie and Sons</td>
      <td>Esequiel Schinner</td>
      <td>HZ-54995</td>
      <td>Belt</td>
      <td>20</td>
      <td>54.67</td>
      <td>1093.40</td>
      <td>2016-02-20 10:43:41</td>
      <td>0.040</td>
      <td>0</td>
      <td>43.7360</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1487</th>
      <td>218667</td>
      <td>Jaskolski-O'Hara</td>
      <td>Trish Deckow</td>
      <td>TL-23025</td>
      <td>Shoes</td>
      <td>19</td>
      <td>45.97</td>
      <td>873.43</td>
      <td>2015-08-12 22:53:38</td>
      <td>0.030</td>
      <td>0</td>
      <td>26.2029</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1488</th>
      <td>575704</td>
      <td>Lindgren, Thompson and Kirlin</td>
      <td>Teagan O'Keefe</td>
      <td>TL-23025</td>
      <td>Shoes</td>
      <td>1</td>
      <td>55.81</td>
      <td>55.81</td>
      <td>2016-02-01 17:50:13</td>
      <td>0.030</td>
      <td>0</td>
      <td>1.6743</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1489</th>
      <td>887145</td>
      <td>Gislason LLC</td>
      <td>Loring Predovic</td>
      <td>ZY-38455</td>
      <td>Shirt</td>
      <td>10</td>
      <td>16.37</td>
      <td>163.70</td>
      <td>2015-10-11 22:11:45</td>
      <td>0.050</td>
      <td>0</td>
      <td>8.1850</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1490</th>
      <td>898496</td>
      <td>Weissnat, Veum and Barton</td>
      <td>Ansley Cummings</td>
      <td>GP-14407</td>
      <td>Belt</td>
      <td>8</td>
      <td>83.37</td>
      <td>666.96</td>
      <td>2016-01-27 03:26:53</td>
      <td>0.030</td>
      <td>0</td>
      <td>20.0088</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1491</th>
      <td>62592</td>
      <td>O'Keefe-Koch</td>
      <td>Shannen Hudson</td>
      <td>TK-29646</td>
      <td>Shoes</td>
      <td>18</td>
      <td>24.92</td>
      <td>448.56</td>
      <td>2016-02-11 17:13:23</td>
      <td>0.030</td>
      <td>0</td>
      <td>13.4568</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1492</th>
      <td>530925</td>
      <td>Purdy and Sons</td>
      <td>Teagan O'Keefe</td>
      <td>HZ-54995</td>
      <td>Belt</td>
      <td>3</td>
      <td>90.33</td>
      <td>270.99</td>
      <td>2015-09-23 07:36:34</td>
      <td>0.030</td>
      <td>0</td>
      <td>8.1297</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1493</th>
      <td>93583</td>
      <td>Hegmann-Howell</td>
      <td>Esequiel Schinner</td>
      <td>GP-14407</td>
      <td>Belt</td>
      <td>5</td>
      <td>45.93</td>
      <td>229.65</td>
      <td>2016-04-13 22:34:56</td>
      <td>0.030</td>
      <td>0</td>
      <td>6.8895</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1494</th>
      <td>14406</td>
      <td>Harber, Lubowitz and Fahey</td>
      <td>Esequiel Schinner</td>
      <td>ZY-38455</td>
      <td>Shirt</td>
      <td>16</td>
      <td>21.60</td>
      <td>345.60</td>
      <td>2015-11-18 06:28:56</td>
      <td>0.050</td>
      <td>0</td>
      <td>17.2800</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1495</th>
      <td>136521</td>
      <td>Labadie and Sons</td>
      <td>Esequiel Schinner</td>
      <td>GP-14407</td>
      <td>Belt</td>
      <td>4</td>
      <td>98.57</td>
      <td>394.28</td>
      <td>2016-06-01 17:28:44</td>
      <td>0.030</td>
      <td>0</td>
      <td>11.8284</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1496</th>
      <td>575704</td>
      <td>Lindgren, Thompson and Kirlin</td>
      <td>Teagan O'Keefe</td>
      <td>TK-29646</td>
      <td>Shoes</td>
      <td>3</td>
      <td>65.16</td>
      <td>195.48</td>
      <td>2016-04-02 16:38:31</td>
      <td>0.030</td>
      <td>0</td>
      <td>5.8644</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1497</th>
      <td>898496</td>
      <td>Weissnat, Veum and Barton</td>
      <td>Ansley Cummings</td>
      <td>EO-54210</td>
      <td>Shirt</td>
      <td>17</td>
      <td>28.10</td>
      <td>477.70</td>
      <td>2015-07-20 19:30:10</td>
      <td>0.050</td>
      <td>0</td>
      <td>23.8850</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1498</th>
      <td>62592</td>
      <td>O'Keefe-Koch</td>
      <td>Shannen Hudson</td>
      <td>FI-01804</td>
      <td>Shirt</td>
      <td>19</td>
      <td>94.96</td>
      <td>1804.24</td>
      <td>2015-10-05 15:55:01</td>
      <td>0.050</td>
      <td>0</td>
      <td>90.2120</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1499</th>
      <td>530925</td>
      <td>Purdy and Sons</td>
      <td>Teagan O'Keefe</td>
      <td>DU-87462</td>
      <td>Shirt</td>
      <td>3</td>
      <td>22.86</td>
      <td>68.58</td>
      <td>2015-10-07 19:49:38</td>
      <td>0.050</td>
      <td>0</td>
      <td>3.4290</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>1500 rows × 13 columns</p>
</div>




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
      <th>account number</th>
      <th>customer name</th>
      <th>sales rep</th>
      <th>sku</th>
      <th>category</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
      <th>commission</th>
      <th>bonus</th>
      <th>comp</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>GP-14407</td>
      <td>Belt</td>
      <td>19</td>
      <td>88.49</td>
      <td>1681.31</td>
      <td>2015-11-17 05:58:34</td>
      <td>0.04</td>
      <td>0</td>
      <td>67.2524</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>FI-01804</td>
      <td>Shirt</td>
      <td>3</td>
      <td>78.07</td>
      <td>234.21</td>
      <td>2016-02-13 04:04:11</td>
      <td>0.05</td>
      <td>0</td>
      <td>11.7105</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>530925</td>
      <td>Purdy and Sons</td>
      <td>Teagan O'Keefe</td>
      <td>EO-54210</td>
      <td>Shirt</td>
      <td>19</td>
      <td>30.21</td>
      <td>573.99</td>
      <td>2015-08-11 12:44:38</td>
      <td>0.05</td>
      <td>0</td>
      <td>28.6995</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14406</td>
      <td>Harber, Lubowitz and Fahey</td>
      <td>Esequiel Schinner</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>12</td>
      <td>90.29</td>
      <td>1083.48</td>
      <td>2016-01-23 02:15:50</td>
      <td>0.05</td>
      <td>0</td>
      <td>54.1740</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>398620</td>
      <td>Brekke Ltd</td>
      <td>Esequiel Schinner</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>5</td>
      <td>72.64</td>
      <td>363.20</td>
      <td>2015-08-10 07:16:03</td>
      <td>0.05</td>
      <td>0</td>
      <td>18.1600</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby("category").agg({"quantity":["count","size"],"unit price":["sum"],"ext price":['mean']})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">quantity</th>
      <th>unit price</th>
      <th>ext price</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>size</th>
      <th>sum</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>category</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Belt</th>
      <td>298</td>
      <td>298</td>
      <td>15754.05</td>
      <td>548.017852</td>
    </tr>
    <tr>
      <th>Shirt</th>
      <td>734</td>
      <td>734</td>
      <td>41696.16</td>
      <td>581.485817</td>
    </tr>
    <tr>
      <th>Shoes</th>
      <td>468</td>
      <td>468</td>
      <td>25524.53</td>
      <td>571.603162</td>
    </tr>
  </tbody>
</table>
</div>



What is nice about the pivot table, is that you have three dataframe parameters, index, columns, and values to adjust, wheras the groupby function only use index and values. 


```python
df.pivot_table(index=["month","sales rep"],columns=["category"], values=["bonus","comp"],aggfunc=[np.sum, np.size],fill_value="No Sale").head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="6" halign="left">sum</th>
      <th colspan="6" halign="left">size</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">bonus</th>
      <th colspan="3" halign="left">comp</th>
      <th colspan="3" halign="left">bonus</th>
      <th colspan="3" halign="left">comp</th>
    </tr>
    <tr>
      <th></th>
      <th>category</th>
      <th>Belt</th>
      <th>Shirt</th>
      <th>Shoes</th>
      <th>Belt</th>
      <th>Shirt</th>
      <th>Shoes</th>
      <th>Belt</th>
      <th>Shirt</th>
      <th>Shoes</th>
      <th>Belt</th>
      <th>Shirt</th>
      <th>Shoes</th>
    </tr>
    <tr>
      <th>month</th>
      <th>sales rep</th>
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
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="7" valign="top">1</th>
      <th>Ansley Cummings</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>84.7025</td>
      <td>34.094</td>
      <td>38.3076</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Beth Skiles</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>24.4992</td>
      <td>35.6795</td>
      <td>17.9136</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Esequiel Schinner</th>
      <td>0</td>
      <td>0</td>
      <td>750</td>
      <td>120.595</td>
      <td>348.077</td>
      <td>1013.11</td>
      <td>3</td>
      <td>13</td>
      <td>8</td>
      <td>3</td>
      <td>13</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Loring Predovic</th>
      <td>0</td>
      <td>0</td>
      <td>250</td>
      <td>37.9589</td>
      <td>252.68</td>
      <td>365.321</td>
      <td>6</td>
      <td>8</td>
      <td>4</td>
      <td>6</td>
      <td>8</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Shannen Hudson</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>116.422</td>
      <td>201.026</td>
      <td>1.6809</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Teagan O'Keefe</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>42.5652</td>
      <td>660.813</td>
      <td>28.7259</td>
      <td>2</td>
      <td>21</td>
      <td>3</td>
      <td>2</td>
      <td>21</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Trish Deckow</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>113.095</td>
      <td>172.417</td>
      <td>20.0688</td>
      <td>7</td>
      <td>8</td>
      <td>3</td>
      <td>7</td>
      <td>8</td>
      <td>3</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2</th>
      <th>Ansley Cummings</th>
      <td>No Sale</td>
      <td>0</td>
      <td>250</td>
      <td>No Sale</td>
      <td>10.0695</td>
      <td>336.991</td>
      <td>No Sale</td>
      <td>1</td>
      <td>4</td>
      <td>No Sale</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Beth Skiles</th>
      <td>No Sale</td>
      <td>0</td>
      <td>1000</td>
      <td>No Sale</td>
      <td>17.809</td>
      <td>1244.55</td>
      <td>No Sale</td>
      <td>2</td>
      <td>5</td>
      <td>No Sale</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Esequiel Schinner</th>
      <td>0</td>
      <td>0</td>
      <td>250</td>
      <td>153.831</td>
      <td>174.106</td>
      <td>413.933</td>
      <td>6</td>
      <td>8</td>
      <td>9</td>
      <td>6</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>Loring Predovic</th>
      <td>0</td>
      <td>0</td>
      <td>1000</td>
      <td>82.4416</td>
      <td>366.779</td>
      <td>1345</td>
      <td>4</td>
      <td>11</td>
      <td>10</td>
      <td>4</td>
      <td>11</td>
      <td>10</td>
    </tr>
    <tr>
      <th>Shannen Hudson</th>
      <td>0</td>
      <td>0</td>
      <td>250</td>
      <td>38.8498</td>
      <td>152.086</td>
      <td>333.645</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



In the privot table below, only certain calculations are applied to certain columns. It is more selective.


```python
# this is whithout brackets, so the type of calculation is not displayed
#df.pivot_table(index=["month","sales rep"],columns=["category"], values=["comp","bonus"],aggfunc={"comp":np.sum, "bonus":np.size},fill_value="No Sale").head(12)
```


```python
df.pivot_table(index=["month","sales rep"],aggfunc={"comp":[np.sum], "bonus":[np.size]},columns=["category"], values=["comp","bonus"],fill_value="No Sale").head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">bonus</th>
      <th colspan="3" halign="left">comp</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">size</th>
      <th colspan="3" halign="left">sum</th>
    </tr>
    <tr>
      <th></th>
      <th>category</th>
      <th>Belt</th>
      <th>Shirt</th>
      <th>Shoes</th>
      <th>Belt</th>
      <th>Shirt</th>
      <th>Shoes</th>
    </tr>
    <tr>
      <th>month</th>
      <th>sales rep</th>
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
      <th rowspan="7" valign="top">1</th>
      <th>Ansley Cummings</th>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>84.7025</td>
      <td>34.094</td>
      <td>38.3076</td>
    </tr>
    <tr>
      <th>Beth Skiles</th>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>24.4992</td>
      <td>35.6795</td>
      <td>17.9136</td>
    </tr>
    <tr>
      <th>Esequiel Schinner</th>
      <td>3</td>
      <td>13</td>
      <td>8</td>
      <td>120.595</td>
      <td>348.077</td>
      <td>1013.11</td>
    </tr>
    <tr>
      <th>Loring Predovic</th>
      <td>6</td>
      <td>8</td>
      <td>4</td>
      <td>37.9589</td>
      <td>252.68</td>
      <td>365.321</td>
    </tr>
    <tr>
      <th>Shannen Hudson</th>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>116.422</td>
      <td>201.026</td>
      <td>1.6809</td>
    </tr>
    <tr>
      <th>Teagan O'Keefe</th>
      <td>2</td>
      <td>21</td>
      <td>3</td>
      <td>42.5652</td>
      <td>660.813</td>
      <td>28.7259</td>
    </tr>
    <tr>
      <th>Trish Deckow</th>
      <td>7</td>
      <td>8</td>
      <td>3</td>
      <td>113.095</td>
      <td>172.417</td>
      <td>20.0688</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2</th>
      <th>Ansley Cummings</th>
      <td>No Sale</td>
      <td>1</td>
      <td>4</td>
      <td>No Sale</td>
      <td>10.0695</td>
      <td>336.991</td>
    </tr>
    <tr>
      <th>Beth Skiles</th>
      <td>No Sale</td>
      <td>2</td>
      <td>5</td>
      <td>No Sale</td>
      <td>17.809</td>
      <td>1244.55</td>
    </tr>
    <tr>
      <th>Esequiel Schinner</th>
      <td>6</td>
      <td>8</td>
      <td>9</td>
      <td>153.831</td>
      <td>174.106</td>
      <td>413.933</td>
    </tr>
    <tr>
      <th>Loring Predovic</th>
      <td>4</td>
      <td>11</td>
      <td>10</td>
      <td>82.4416</td>
      <td>366.779</td>
      <td>1345</td>
    </tr>
    <tr>
      <th>Shannen Hudson</th>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>38.8498</td>
      <td>152.086</td>
      <td>333.645</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_pivot = df.pivot_table(index=["month","sales rep"],aggfunc={"comp":[np.sum], "bonus":[np.size]},columns=["category"], values=["comp","bonus"],fill_value="No Sale")
```

You can now if you feel the need to, do some querying 


```python
df_pivot.query("month == [1]")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">bonus</th>
      <th colspan="3" halign="left">comp</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">size</th>
      <th colspan="3" halign="left">sum</th>
    </tr>
    <tr>
      <th></th>
      <th>category</th>
      <th>Belt</th>
      <th>Shirt</th>
      <th>Shoes</th>
      <th>Belt</th>
      <th>Shirt</th>
      <th>Shoes</th>
    </tr>
    <tr>
      <th>month</th>
      <th>sales rep</th>
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
      <th rowspan="7" valign="top">1</th>
      <th>Ansley Cummings</th>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>84.7025</td>
      <td>34.094</td>
      <td>38.3076</td>
    </tr>
    <tr>
      <th>Beth Skiles</th>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>24.4992</td>
      <td>35.6795</td>
      <td>17.9136</td>
    </tr>
    <tr>
      <th>Esequiel Schinner</th>
      <td>3</td>
      <td>13</td>
      <td>8</td>
      <td>120.595</td>
      <td>348.077</td>
      <td>1013.11</td>
    </tr>
    <tr>
      <th>Loring Predovic</th>
      <td>6</td>
      <td>8</td>
      <td>4</td>
      <td>37.9589</td>
      <td>252.68</td>
      <td>365.321</td>
    </tr>
    <tr>
      <th>Shannen Hudson</th>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>116.422</td>
      <td>201.026</td>
      <td>1.6809</td>
    </tr>
    <tr>
      <th>Teagan O'Keefe</th>
      <td>2</td>
      <td>21</td>
      <td>3</td>
      <td>42.5652</td>
      <td>660.813</td>
      <td>28.7259</td>
    </tr>
    <tr>
      <th>Trish Deckow</th>
      <td>7</td>
      <td>8</td>
      <td>3</td>
      <td>113.095</td>
      <td>172.417</td>
      <td>20.0688</td>
    </tr>
  </tbody>
</table>
</div>



This is another way to do it, I find it more reliable 


```python
df_pivot[df_pivot.index.get_level_values(0).isin([1])]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">bonus</th>
      <th colspan="3" halign="left">comp</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">size</th>
      <th colspan="3" halign="left">sum</th>
    </tr>
    <tr>
      <th></th>
      <th>category</th>
      <th>Belt</th>
      <th>Shirt</th>
      <th>Shoes</th>
      <th>Belt</th>
      <th>Shirt</th>
      <th>Shoes</th>
    </tr>
    <tr>
      <th>month</th>
      <th>sales rep</th>
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
      <th rowspan="7" valign="top">1</th>
      <th>Ansley Cummings</th>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>84.7025</td>
      <td>34.094</td>
      <td>38.3076</td>
    </tr>
    <tr>
      <th>Beth Skiles</th>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>24.4992</td>
      <td>35.6795</td>
      <td>17.9136</td>
    </tr>
    <tr>
      <th>Esequiel Schinner</th>
      <td>3</td>
      <td>13</td>
      <td>8</td>
      <td>120.595</td>
      <td>348.077</td>
      <td>1013.11</td>
    </tr>
    <tr>
      <th>Loring Predovic</th>
      <td>6</td>
      <td>8</td>
      <td>4</td>
      <td>37.9589</td>
      <td>252.68</td>
      <td>365.321</td>
    </tr>
    <tr>
      <th>Shannen Hudson</th>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>116.422</td>
      <td>201.026</td>
      <td>1.6809</td>
    </tr>
    <tr>
      <th>Teagan O'Keefe</th>
      <td>2</td>
      <td>21</td>
      <td>3</td>
      <td>42.5652</td>
      <td>660.813</td>
      <td>28.7259</td>
    </tr>
    <tr>
      <th>Trish Deckow</th>
      <td>7</td>
      <td>8</td>
      <td>3</td>
      <td>113.095</td>
      <td>172.417</td>
      <td>20.0688</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_pivot[df_pivot.index.get_level_values(1).isin(["Ansley Cummings"])]

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">bonus</th>
      <th colspan="3" halign="left">comp</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">size</th>
      <th colspan="3" halign="left">sum</th>
    </tr>
    <tr>
      <th></th>
      <th>category</th>
      <th>Belt</th>
      <th>Shirt</th>
      <th>Shoes</th>
      <th>Belt</th>
      <th>Shirt</th>
      <th>Shoes</th>
    </tr>
    <tr>
      <th>month</th>
      <th>sales rep</th>
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
      <th>1</th>
      <th>Ansley Cummings</th>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>84.7025</td>
      <td>34.094</td>
      <td>38.3076</td>
    </tr>
    <tr>
      <th>2</th>
      <th>Ansley Cummings</th>
      <td>No Sale</td>
      <td>1</td>
      <td>4</td>
      <td>No Sale</td>
      <td>10.0695</td>
      <td>336.991</td>
    </tr>
    <tr>
      <th>3</th>
      <th>Ansley Cummings</th>
      <td>No Sale</td>
      <td>2</td>
      <td>2</td>
      <td>No Sale</td>
      <td>48.57</td>
      <td>313.561</td>
    </tr>
    <tr>
      <th>4</th>
      <th>Ansley Cummings</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>50.112</td>
      <td>62.678</td>
      <td>10.4796</td>
    </tr>
    <tr>
      <th>5</th>
      <th>Ansley Cummings</th>
      <td>1</td>
      <td>3</td>
      <td>No Sale</td>
      <td>25.172</td>
      <td>76.116</td>
      <td>No Sale</td>
    </tr>
    <tr>
      <th>6</th>
      <th>Ansley Cummings</th>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>25.9473</td>
      <td>54.596</td>
      <td>309.125</td>
    </tr>
    <tr>
      <th>7</th>
      <th>Ansley Cummings</th>
      <td>No Sale</td>
      <td>2</td>
      <td>2</td>
      <td>No Sale</td>
      <td>36.5295</td>
      <td>21.903</td>
    </tr>
    <tr>
      <th>8</th>
      <th>Ansley Cummings</th>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>51.198</td>
      <td>90.2805</td>
      <td>29.8443</td>
    </tr>
    <tr>
      <th>9</th>
      <th>Ansley Cummings</th>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>39.3054</td>
      <td>49.317</td>
      <td>21.0768</td>
    </tr>
    <tr>
      <th>10</th>
      <th>Ansley Cummings</th>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>20.1152</td>
      <td>154.215</td>
      <td>336.159</td>
    </tr>
    <tr>
      <th>11</th>
      <th>Ansley Cummings</th>
      <td>2</td>
      <td>2</td>
      <td>No Sale</td>
      <td>8.0844</td>
      <td>72.3245</td>
      <td>No Sale</td>
    </tr>
    <tr>
      <th>12</th>
      <th>Ansley Cummings</th>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>73.1176</td>
      <td>164.834</td>
      <td>50.8647</td>
    </tr>
  </tbody>
</table>
</div>




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
      <th>account number</th>
      <th>customer name</th>
      <th>sales rep</th>
      <th>sku</th>
      <th>category</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
      <th>commission</th>
      <th>bonus</th>
      <th>comp</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>GP-14407</td>
      <td>Belt</td>
      <td>19</td>
      <td>88.49</td>
      <td>1681.31</td>
      <td>2015-11-17 05:58:34</td>
      <td>0.04</td>
      <td>0</td>
      <td>67.2524</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>680916</td>
      <td>Mueller and Sons</td>
      <td>Loring Predovic</td>
      <td>FI-01804</td>
      <td>Shirt</td>
      <td>3</td>
      <td>78.07</td>
      <td>234.21</td>
      <td>2016-02-13 04:04:11</td>
      <td>0.05</td>
      <td>0</td>
      <td>11.7105</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>530925</td>
      <td>Purdy and Sons</td>
      <td>Teagan O'Keefe</td>
      <td>EO-54210</td>
      <td>Shirt</td>
      <td>19</td>
      <td>30.21</td>
      <td>573.99</td>
      <td>2015-08-11 12:44:38</td>
      <td>0.05</td>
      <td>0</td>
      <td>28.6995</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14406</td>
      <td>Harber, Lubowitz and Fahey</td>
      <td>Esequiel Schinner</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>12</td>
      <td>90.29</td>
      <td>1083.48</td>
      <td>2016-01-23 02:15:50</td>
      <td>0.05</td>
      <td>0</td>
      <td>54.1740</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>398620</td>
      <td>Brekke Ltd</td>
      <td>Esequiel Schinner</td>
      <td>NZ-99565</td>
      <td>Shirt</td>
      <td>5</td>
      <td>72.64</td>
      <td>363.20</td>
      <td>2015-08-10 07:16:03</td>
      <td>0.05</td>
      <td>0</td>
      <td>18.1600</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>


