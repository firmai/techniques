
# A/B Testing for ShoeFly.com

Our favorite online shoe store, ShoeFly.com is performing an A/B Test. They have two different versions of an ad, which they have placed in emails, as well as in banner ads on Facebook, Twitter, and Google. They want to know how the two ads are performing on each of the different platforms on each day of the week. Help them analyze the data using aggregate measures.


```python
import pandas as pd
ad_clicks = pd.read_csv('ad_clicks.csv')
ad_clicks.head()
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
      <th>user_id</th>
      <th>utm_source</th>
      <th>day</th>
      <th>ad_click_timestamp</th>
      <th>experimental_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>008b7c6c-7272-471e-b90e-930d548bd8d7</td>
      <td>google</td>
      <td>6 - Saturday</td>
      <td>7:18</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>009abb94-5e14-4b6c-bb1c-4f4df7aa7557</td>
      <td>facebook</td>
      <td>7 - Sunday</td>
      <td>NaN</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00f5d532-ed58-4570-b6d2-768df5f41aed</td>
      <td>twitter</td>
      <td>2 - Tuesday</td>
      <td>NaN</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>011adc64-0f44-4fd9-a0bb-f1506d2ad439</td>
      <td>google</td>
      <td>2 - Tuesday</td>
      <td>NaN</td>
      <td>B</td>
    </tr>
    <tr>
      <th>4</th>
      <td>012137e6-7ae7-4649-af68-205b4702169c</td>
      <td>facebook</td>
      <td>7 - Sunday</td>
      <td>NaN</td>
      <td>B</td>
    </tr>
  </tbody>
</table>
</div>



Your manager wants to know which ad platform is getting you the most views.

How many views (i.e., rows of the table) came from each utm_source?


```python
count_rows_by_source = ad_clicks.groupby('utm_source').user_id.count().reset_index()
```

If the column ad_click_timestamp is not null, then someone actually clicked on the ad that was displayed.

Create a new column called is_click, which is True if ad_click_timestamp is not null and False otherwise.


```python
ad_clicks['is_click'] = ~ad_clicks\ # The ~ is a NOT operator, and isnull() tests whether or not the value of ad_click_timestamp is null.
   .ad_click_timestamp.isnull()
```

We want to know the percent of people who clicked on ads from each utm_source.

Start by grouping by utm_source and is_click and counting the number of user_id's in each of those groups. Save your answer to the variable clicks_by_source.


```python
clicks_by_source = ad_clicks\
   .groupby(['utm_source',
             'is_click'])\
   .user_id.count()\
   .reset_index()
```

Now let's pivot the data so that the columns are is_click (either True or False), the index is utm_source, and the values are user_id.

Save your results to the variable clicks_pivot.


```python
clicks_pivot = clicks_by_source.pivot(columns='is_click',values='user_id',index='utm_source').reset_index()
```

Create a new column in clicks_pivot called percent_clicked which is equal to the percent of users who clicked on the ad from each utm_source.

Was there a difference in click rates for each source?


```python
clicks_pivot['percent_clicked'] = clicks_pivot[True] / (clicks_pivot[True] + clicks_pivot[False])
clicks_pivot
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
      <th>is_click</th>
      <th>utm_source</th>
      <th>False</th>
      <th>True</th>
      <th>percent_clicked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>email</td>
      <td>175</td>
      <td>80</td>
      <td>0.313725</td>
    </tr>
    <tr>
      <th>1</th>
      <td>facebook</td>
      <td>324</td>
      <td>180</td>
      <td>0.357143</td>
    </tr>
    <tr>
      <th>2</th>
      <td>google</td>
      <td>441</td>
      <td>239</td>
      <td>0.351471</td>
    </tr>
    <tr>
      <th>3</th>
      <td>twitter</td>
      <td>149</td>
      <td>66</td>
      <td>0.306977</td>
    </tr>
  </tbody>
</table>
</div>



The column experimental_group tells us whether the user was shown Ad A or Ad B.

Were approximately the same number of people shown both adds?


```python
ad_clicks.groupby('experimental_group').count().reset_index()
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
      <th>experimental_group</th>
      <th>user_id</th>
      <th>utm_source</th>
      <th>day</th>
      <th>ad_click_timestamp</th>
      <th>is_click</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>827</td>
      <td>827</td>
      <td>827</td>
      <td>310</td>
      <td>827</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>827</td>
      <td>827</td>
      <td>827</td>
      <td>255</td>
      <td>827</td>
    </tr>
  </tbody>
</table>
</div>



Using the column is_click that we defined earlier, check to see if a greater percentage of users clicked on Ad A or Ad B.



```python
percent_a_b = ad_clicks.groupby(['experimental_group', 'is_click']).count().reset_index()
percent_a_b.pivot(columns='experimental_group',values='is_click',index='user_id').reset_index()
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
      <th>experimental_group</th>
      <th>user_id</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>255</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>310</td>
      <td>True</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>517</td>
      <td>False</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>572</td>
      <td>None</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



The Product Manager for the A/B test thinks that the clicks might have changed by day of the week.

Start by creating two DataFrames: a_clicks and b_clicks, which contain only the results for A group and B group, respectively.


```python
a_clicks = ad_clicks[ad_clicks['experimental_group'] == 'A']
b_clicks = ad_clicks[ad_clicks['experimental_group'] == 'B']
```

For each group (a_clicks and b_clicks), calculate the percent of users who clicked on the ad by day.


```python
a_percent_by_day = a_clicks.groupby(['is_click', 'day']).count().reset_index()

```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-34-d5ce7b00f0ff> in <module>()
          1 a_percent_by_day = a_clicks.groupby(['is_click', 'day']).count().reset_index()
    ----> 2 a_pivot = a_percent_by_day.pivot(columns='is_click',values='day',index='user_id').reset_index()
    

    c:\users\brocd8s\appdata\local\programs\python\python36-32\lib\site-packages\pandas\core\frame.py in pivot(self, index, columns, values)
       4380         """
       4381         from pandas.core.reshape.reshape import pivot
    -> 4382         return pivot(self, index=index, columns=columns, values=values)
       4383 
       4384     _shared_docs['pivot_table'] = """


    c:\users\brocd8s\appdata\local\programs\python\python36-32\lib\site-packages\pandas\core\reshape\reshape.py in pivot(self, index, columns, values)
        387         indexed = Series(self[values].values,
        388                          index=MultiIndex.from_arrays([index, self[columns]]))
    --> 389         return indexed.unstack(columns)
        390 
        391 


    c:\users\brocd8s\appdata\local\programs\python\python36-32\lib\site-packages\pandas\core\series.py in unstack(self, level, fill_value)
       2222         """
       2223         from pandas.core.reshape.reshape import unstack
    -> 2224         return unstack(self, level, fill_value)
       2225 
       2226     # ----------------------------------------------------------------------


    c:\users\brocd8s\appdata\local\programs\python\python36-32\lib\site-packages\pandas\core\reshape\reshape.py in unstack(obj, level, fill_value)
        472     else:
        473         unstacker = _Unstacker(obj.values, obj.index, level=level,
    --> 474                                fill_value=fill_value)
        475         return unstacker.get_result()
        476 


    c:\users\brocd8s\appdata\local\programs\python\python36-32\lib\site-packages\pandas\core\reshape\reshape.py in __init__(self, values, index, level, value_columns, fill_value)
        114 
        115         self._make_sorted_values_labels()
    --> 116         self._make_selectors()
        117 
        118     def _make_sorted_values_labels(self):


    c:\users\brocd8s\appdata\local\programs\python\python36-32\lib\site-packages\pandas\core\reshape\reshape.py in _make_selectors(self)
        152 
        153         if mask.sum() < len(self.index):
    --> 154             raise ValueError('Index contains duplicate entries, '
        155                              'cannot reshape')
        156 


    ValueError: Index contains duplicate entries, cannot reshape


Compare the results for A and B. What happened over the course of the week?

Do you recommend that your company use Ad A or Ad B?
