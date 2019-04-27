

```python
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.models.formatters import NumeralTickFormatter
import pandas as pd
```


```python
output_notebook()
```



    <div class="bk-root">
        <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="0aa7ba42-8fd8-4e98-a271-d0a17bbf6005">Loading BokehJS ...</span>
    </div>





```python
# Create the initial dataframe
index = ['sales','returns','credit fees','rebates','late charges','shipping']
data = {'amount': [350000,-30000,-7500,-25000,95000,-7000]}
df = pd.DataFrame(data=data,index=index)

# Determine the total net value by adding the start and all additional transactions
net = df['amount'].sum()
```


```python
df
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
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sales</th>
      <td>350000</td>
    </tr>
    <tr>
      <th>returns</th>
      <td>-30000</td>
    </tr>
    <tr>
      <th>credit fees</th>
      <td>-7500</td>
    </tr>
    <tr>
      <th>rebates</th>
      <td>-25000</td>
    </tr>
    <tr>
      <th>late charges</th>
      <td>95000</td>
    </tr>
    <tr>
      <th>shipping</th>
      <td>-7000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create additional columns that we will use to build the waterfall
df['running_total'] = df['amount'].cumsum()
df['y_start'] = df['running_total'] - df['amount']

# Where do we want to place the label
df['label_pos'] = df['running_total']
```


```python
df
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
      <th>amount</th>
      <th>running_total</th>
      <th>y_start</th>
      <th>label_pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sales</th>
      <td>350000</td>
      <td>350000</td>
      <td>0</td>
      <td>350000</td>
    </tr>
    <tr>
      <th>returns</th>
      <td>-30000</td>
      <td>320000</td>
      <td>350000</td>
      <td>320000</td>
    </tr>
    <tr>
      <th>credit fees</th>
      <td>-7500</td>
      <td>312500</td>
      <td>320000</td>
      <td>312500</td>
    </tr>
    <tr>
      <th>rebates</th>
      <td>-25000</td>
      <td>287500</td>
      <td>312500</td>
      <td>287500</td>
    </tr>
    <tr>
      <th>late charges</th>
      <td>95000</td>
      <td>382500</td>
      <td>287500</td>
      <td>382500</td>
    </tr>
    <tr>
      <th>shipping</th>
      <td>-7000</td>
      <td>375500</td>
      <td>382500</td>
      <td>375500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We need to have a net column at the end with the totals and a full bar
df_net = pd.DataFrame.from_records([(net, net, 0, net)], 
                                   columns=['amount', 'running_total', 'y_start', 'label_pos'],
                                   index=["net"])
df = df.append(df_net)
```


```python
df
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
      <th>amount</th>
      <th>running_total</th>
      <th>y_start</th>
      <th>label_pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sales</th>
      <td>350000</td>
      <td>350000</td>
      <td>0</td>
      <td>350000</td>
    </tr>
    <tr>
      <th>returns</th>
      <td>-30000</td>
      <td>320000</td>
      <td>350000</td>
      <td>320000</td>
    </tr>
    <tr>
      <th>credit fees</th>
      <td>-7500</td>
      <td>312500</td>
      <td>320000</td>
      <td>312500</td>
    </tr>
    <tr>
      <th>rebates</th>
      <td>-25000</td>
      <td>287500</td>
      <td>312500</td>
      <td>287500</td>
    </tr>
    <tr>
      <th>late charges</th>
      <td>95000</td>
      <td>382500</td>
      <td>287500</td>
      <td>382500</td>
    </tr>
    <tr>
      <th>shipping</th>
      <td>-7000</td>
      <td>375500</td>
      <td>382500</td>
      <td>375500</td>
    </tr>
    <tr>
      <th>net</th>
      <td>375500</td>
      <td>375500</td>
      <td>0</td>
      <td>375500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We want to color the positive values gray and the negative red
df['color'] = 'grey'
df.loc[df.amount < 0, 'color'] = 'red'

# The 10000 factor is used to make the text positioned correctly.
# You will need to modify if the values are significantly different
df.loc[df.amount < 0, 'label_pos'] = df.label_pos - 10000
df["bar_label"] = df["amount"].map('{:,.0f}'.format)
```


```python
df
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
      <th>amount</th>
      <th>running_total</th>
      <th>y_start</th>
      <th>label_pos</th>
      <th>color</th>
      <th>bar_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sales</th>
      <td>350000</td>
      <td>350000</td>
      <td>0</td>
      <td>350000</td>
      <td>grey</td>
      <td>350,000</td>
    </tr>
    <tr>
      <th>returns</th>
      <td>-30000</td>
      <td>320000</td>
      <td>350000</td>
      <td>310000</td>
      <td>red</td>
      <td>-30,000</td>
    </tr>
    <tr>
      <th>credit fees</th>
      <td>-7500</td>
      <td>312500</td>
      <td>320000</td>
      <td>302500</td>
      <td>red</td>
      <td>-7,500</td>
    </tr>
    <tr>
      <th>rebates</th>
      <td>-25000</td>
      <td>287500</td>
      <td>312500</td>
      <td>277500</td>
      <td>red</td>
      <td>-25,000</td>
    </tr>
    <tr>
      <th>late charges</th>
      <td>95000</td>
      <td>382500</td>
      <td>287500</td>
      <td>382500</td>
      <td>grey</td>
      <td>95,000</td>
    </tr>
    <tr>
      <th>shipping</th>
      <td>-7000</td>
      <td>375500</td>
      <td>382500</td>
      <td>365500</td>
      <td>red</td>
      <td>-7,000</td>
    </tr>
    <tr>
      <th>net</th>
      <td>375500</td>
      <td>375500</td>
      <td>0</td>
      <td>375500</td>
      <td>grey</td>
      <td>375,500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Build the Bokeh figure

# Limit the tools to only these three
TOOLS = "box_zoom,reset,save"

# Build the source data off the df dataframe
source = ColumnDataSource(df)

# Create the figure and assign range values that look good for the data set
p = figure(tools=TOOLS, x_range=list(df.index), y_range=(0, net+40000), plot_width=800, title = "Sales Waterfall")
p.grid.grid_line_alpha=0.3

# Add the segments
p.segment(x0='index', y0='y_start', x1="index", y1='running_total', source=source, color="color", line_width=55)

# Format the y-axis as dollars
p.yaxis[0].formatter = NumeralTickFormatter(format="($ 0 a)")
p.xaxis.axis_label = "Transactions"

# Add the labels
labels = LabelSet(x='index', y='label_pos', text='bar_label', text_font_size="8pt", level='glyph',
         x_offset=-20, y_offset=0, source=source)
p.add_layout(labels)
```


```python
show(p)
```

    /Users/dereksnow/anaconda/envs/py36/lib/python3.6/site-packages/bokeh/core/json_encoder.py:80: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      elif np.issubdtype(type(obj), np.float):




<div class="bk-root">
    <div class="bk-plotdiv" id="ada04326-3b67-4b75-880a-6e993656cf7e"></div>
</div>



