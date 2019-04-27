

```python
import pandas
df_hist = pandas.read_excel("soda_sales_historical_data.xlsx")
df_hist[:5]

df_hist[df_hist["Product"]=="11 Down"]

from pandas import DataFrame, get_dummies
categorical_columns = ['Product','Easter Included','Super Bowl Included', 
                       'Christmas Included', 'Other Holiday']
df_hist = get_dummies(df_hist, prefix={k:"dmy_%s"%k for k in categorical_columns},
                      columns = list(categorical_columns))
df_hist[:5]
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
      <th>Sales</th>
      <th>Cost Per Unit</th>
      <th>4 Wk Avg Temp</th>
      <th>4 Wk Avg Humidity</th>
      <th>Sales M-1 weeks</th>
      <th>Sales M-2 weeks</th>
      <th>Sales M-3 weeks</th>
      <th>Sales M-4 Weeks</th>
      <th>Sales M-5 weeks</th>
      <th>dmy_Product_11 Down</th>
      <th>...</th>
      <th>dmy_Product_Koala Kola</th>
      <th>dmy_Product_Mr. Popper</th>
      <th>dmy_Product_Popsi Kola</th>
      <th>dmy_Easter Included_No</th>
      <th>dmy_Easter Included_Yes</th>
      <th>dmy_Super Bowl Included_No</th>
      <th>dmy_Super Bowl Included_Yes</th>
      <th>dmy_Christmas Included_No</th>
      <th>dmy_Christmas Included_Yes</th>
      <th>dmy_Other Holiday_No</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>51.9</td>
      <td>1.6625</td>
      <td>80.69</td>
      <td>69.19</td>
      <td>17.0</td>
      <td>22.4</td>
      <td>13.5</td>
      <td>14.5</td>
      <td>28.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55.8</td>
      <td>2.2725</td>
      <td>80.69</td>
      <td>69.19</td>
      <td>2.4</td>
      <td>2.2</td>
      <td>2.0</td>
      <td>1.4</td>
      <td>0.5</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3385.6</td>
      <td>1.3475</td>
      <td>80.69</td>
      <td>69.19</td>
      <td>301.8</td>
      <td>188.8</td>
      <td>101.4</td>
      <td>81.6</td>
      <td>213.8</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>63.5</td>
      <td>1.6600</td>
      <td>80.69</td>
      <td>69.19</td>
      <td>73.8</td>
      <td>69.4</td>
      <td>72.8</td>
      <td>75.4</td>
      <td>57.4</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>181.1</td>
      <td>1.8725</td>
      <td>80.69</td>
      <td>69.19</td>
      <td>23.1</td>
      <td>22.6</td>
      <td>22.1</td>
      <td>19.9</td>
      <td>23.2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn import model_selection
experiments = {"Algorithm":["Ordinary Least Squares", "Regression Tree", 
                            "Big Random Forest", "Random Forest", 
                            "Bagging"], 
               "Objects" : [lambda : LinearRegression(), 
                            lambda : DecisionTreeRegressor(), 
                            lambda : RandomForestRegressor(n_estimators=100), 
                            lambda : RandomForestRegressor(), 
                            lambda : BaggingRegressor()], 
               "Predictions":[[] for _ in range(5)]}
actuals = []
```


```python
from sklearn.model_selection import train_test_split
for _ in range (4):
    train_X, test_X, train_y, test_y = (
        train_test_split(df_hist.drop("Sales", axis=1), 
                         df_hist["Sales"], test_size=0.25))
    for i, obj_factory in enumerate(experiments["Objects"]):
        obj = obj_factory()
        obj.fit(y=train_y,X=train_X)
        experiments["Predictions"][i] += list(obj.predict(test_X))
    actuals += list(test_y)
actuals = pandas.Series(actuals)
experiments["Predictions"] = list(map(pandas.Series, experiments["Predictions"]))
```


```python
experiments["Results"] = []
for o in experiments["Objects"]:
    experiments["Results"].append(
        model_selection.cross_val_score(o(), y=df_hist['Sales'], 
                                        X=df_hist.drop("Sales", axis=1),
                                        cv=5).mean())
DataFrame(experiments).drop(["Objects", "Predictions"], 
                            axis=1).set_index("Algorithm")

fitted = (experiments["Objects"]
          [experiments["Algorithm"].index("Big Random Forest")]().
          fit(y=df_hist["Sales"], X=df_hist.drop("Sales", axis=1)))
```


```python
df_superbowl_original = pandas.read_excel("super_bowl_promotion_data.xlsx")
df_superbowl = get_dummies(df_superbowl_original, 
                           prefix={k:"dmy_%s"%k for k in categorical_columns},
                           columns = list(categorical_columns))
assert "Sales" not in df_superbowl.columns 
assert {"Sales"}.union(df_superbowl.columns).issubset(set(df_hist.columns))
len(df_superbowl)
```




    36




```python
for fld in set(df_hist.columns).difference(df_superbowl.columns, {"Sales"}):
    assert fld.startswith("dmy_")
    df_superbowl[fld] = 0
    
df_superbowl = df_superbowl[list(df_hist.drop("Sales", axis=1).columns)]

predicted = fitted.predict(df_superbowl)

```


```python
from __future__ import print_function
from ortools.constraint_solver import pywrapcp as cp

soda_family = {'11 Down': 'Clear', 'AB Root Beer': 'Dark', 
               'Alpine Stream': 'Clear', 'Bright': 'Clear', 
               'Crisp Clear': 'Clear', 'DC Kola': 'Dark',
               'Koala Kola': 'Dark', 'Mr. Popper': 'Dark', 
               'Popsi Kola': 'Dark'}
family  = set(soda_family[j] for j in soda_family)
soda    = set(j for j in soda_family)
max_prom = {f:2 for f in family}
product_prices = set(forecast_sales.index.values)
normal_price = {b:0 for b in soda}
for b,p in product_prices:
    normal_price[b] = max(normal_price[b],p)
```


```python
forecast_sales = df_superbowl_original[["Product", "Cost Per Unit"]].copy()
forecast_sales["Sales"] = predicted

forecast_sales = forecast_sales.groupby(['Product','Cost Per Unit']).mean()

normal_price_2 = dict(forecast_sales.reset_index()[["Product","Cost Per Unit"]].groupby("Product").max()["Cost Per Unit"])



maxer = forecast_sales.reset_index().groupby("Product").max()


## This is to get the sales value of the max cost

maxer["Sales"] = forecast_sales[forecast_sales.index.isin(maxer.reset_index().set_index(["Product","Cost Per Unit"]).index)]["Sales"].values
```


```python
maxer
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
      <th>Cost Per Unit</th>
      <th>Sales</th>
    </tr>
    <tr>
      <th>Product</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11 Down</th>
      <td>1.5600</td>
      <td>173.626</td>
    </tr>
    <tr>
      <th>AB Root Beer</th>
      <td>3.8425</td>
      <td>376.175</td>
    </tr>
    <tr>
      <th>Alpine Stream</th>
      <td>2.2275</td>
      <td>156.269</td>
    </tr>
    <tr>
      <th>Bright</th>
      <td>1.2900</td>
      <td>2656.430</td>
    </tr>
    <tr>
      <th>Crisp Clear</th>
      <td>1.4700</td>
      <td>140.921</td>
    </tr>
    <tr>
      <th>DC Kola</th>
      <td>1.9325</td>
      <td>501.308</td>
    </tr>
    <tr>
      <th>Koala Kola</th>
      <td>2.5650</td>
      <td>1423.454</td>
    </tr>
    <tr>
      <th>Mr. Popper</th>
      <td>2.9850</td>
      <td>38.421</td>
    </tr>
    <tr>
      <th>Popsi Kola</th>
      <td>1.7500</td>
      <td>145.957</td>
    </tr>
  </tbody>
</table>
</div>




```python
import pandas as pd
forecast_sales_r =forecast_sales.reset_index()
bat = pd.DataFrame()
for product in maxer.index:
    rat = forecast_sales_r[forecast_sales_r["Product"]==product]["Cost Per Unit"]-maxer.loc[product,"Cost Per Unit"]
    rat = rat.to_frame()
    bat = pd.concat((bat,rat))

forecast_sales_r["marginal_investment"] =  bat.values
```


```python
import pandas as pd

bat = pd.DataFrame()
for product in maxer.index:
    rat = forecast_sales_r[forecast_sales_r["Product"]==product]["Sales"]-maxer.loc[product,"Sales"]
    rat = rat.to_frame()
    bat = pd.concat((bat,rat))
forecast_sales_r["extra_sales"] =  bat.values
```


```python
forecast_sales_r
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
      <th>Product</th>
      <th>Cost Per Unit</th>
      <th>Sales</th>
      <th>marginal_investment</th>
      <th>extra_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11 Down</td>
      <td>1.4550</td>
      <td>248.407</td>
      <td>-0.1050</td>
      <td>74.781</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11 Down</td>
      <td>1.5125</td>
      <td>239.704</td>
      <td>-0.0475</td>
      <td>66.078</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11 Down</td>
      <td>1.5375</td>
      <td>211.602</td>
      <td>-0.0225</td>
      <td>37.976</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11 Down</td>
      <td>1.5600</td>
      <td>173.626</td>
      <td>0.0000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AB Root Beer</td>
      <td>3.7300</td>
      <td>384.838</td>
      <td>-0.1125</td>
      <td>8.663</td>
    </tr>
    <tr>
      <th>5</th>
      <td>AB Root Beer</td>
      <td>3.7700</td>
      <td>389.141</td>
      <td>-0.0725</td>
      <td>12.966</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AB Root Beer</td>
      <td>3.8125</td>
      <td>379.363</td>
      <td>-0.0300</td>
      <td>3.188</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AB Root Beer</td>
      <td>3.8425</td>
      <td>376.175</td>
      <td>0.0000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Alpine Stream</td>
      <td>1.9975</td>
      <td>201.513</td>
      <td>-0.2300</td>
      <td>45.244</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Alpine Stream</td>
      <td>2.1375</td>
      <td>174.656</td>
      <td>-0.0900</td>
      <td>18.387</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Alpine Stream</td>
      <td>2.1500</td>
      <td>176.016</td>
      <td>-0.0775</td>
      <td>19.747</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Alpine Stream</td>
      <td>2.2275</td>
      <td>156.269</td>
      <td>0.0000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Bright</td>
      <td>1.2725</td>
      <td>3790.724</td>
      <td>-0.0175</td>
      <td>1134.294</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Bright</td>
      <td>1.2825</td>
      <td>2802.034</td>
      <td>-0.0075</td>
      <td>145.604</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Bright</td>
      <td>1.2900</td>
      <td>2656.430</td>
      <td>0.0000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Crisp Clear</td>
      <td>1.3125</td>
      <td>184.459</td>
      <td>-0.1575</td>
      <td>43.538</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Crisp Clear</td>
      <td>1.3425</td>
      <td>173.047</td>
      <td>-0.1275</td>
      <td>32.126</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Crisp Clear</td>
      <td>1.4275</td>
      <td>138.466</td>
      <td>-0.0425</td>
      <td>-2.455</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Crisp Clear</td>
      <td>1.4700</td>
      <td>140.921</td>
      <td>0.0000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>DC Kola</td>
      <td>1.8900</td>
      <td>936.894</td>
      <td>-0.0425</td>
      <td>435.586</td>
    </tr>
    <tr>
      <th>20</th>
      <td>DC Kola</td>
      <td>1.9150</td>
      <td>568.984</td>
      <td>-0.0175</td>
      <td>67.676</td>
    </tr>
    <tr>
      <th>21</th>
      <td>DC Kola</td>
      <td>1.9250</td>
      <td>512.329</td>
      <td>-0.0075</td>
      <td>11.021</td>
    </tr>
    <tr>
      <th>22</th>
      <td>DC Kola</td>
      <td>1.9325</td>
      <td>501.308</td>
      <td>0.0000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Koala Kola</td>
      <td>2.4825</td>
      <td>1394.552</td>
      <td>-0.0825</td>
      <td>-28.902</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Koala Kola</td>
      <td>2.5350</td>
      <td>1408.344</td>
      <td>-0.0300</td>
      <td>-15.110</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Koala Kola</td>
      <td>2.5600</td>
      <td>1426.688</td>
      <td>-0.0050</td>
      <td>3.234</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Koala Kola</td>
      <td>2.5650</td>
      <td>1423.454</td>
      <td>0.0000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Mr. Popper</td>
      <td>2.8475</td>
      <td>41.408</td>
      <td>-0.1375</td>
      <td>2.987</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Mr. Popper</td>
      <td>2.8925</td>
      <td>41.404</td>
      <td>-0.0925</td>
      <td>2.983</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Mr. Popper</td>
      <td>2.8950</td>
      <td>41.404</td>
      <td>-0.0900</td>
      <td>2.983</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Mr. Popper</td>
      <td>2.9000</td>
      <td>41.382</td>
      <td>-0.0850</td>
      <td>2.961</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Mr. Popper</td>
      <td>2.9850</td>
      <td>38.421</td>
      <td>0.0000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Popsi Kola</td>
      <td>1.6725</td>
      <td>156.135</td>
      <td>-0.0775</td>
      <td>10.178</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Popsi Kola</td>
      <td>1.7125</td>
      <td>145.706</td>
      <td>-0.0375</td>
      <td>-0.251</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Popsi Kola</td>
      <td>1.7275</td>
      <td>147.310</td>
      <td>-0.0225</td>
      <td>1.353</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Popsi Kola</td>
      <td>1.7500</td>
      <td>145.957</td>
      <td>0.0000</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
for j in range(num_tasks):
    ## This in effect sets constraints on the domain
    t.append(solver.IntVar(0, 1, "x[%i,%i]" % (i, j)))
```


```python
forecast_sales = forecast_sales.reset_index()
```


```python
forecast_sales
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
      <th>Product</th>
      <th>Cost Per Unit</th>
      <th>Sales</th>
      <th>net_margin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11 Down</td>
      <td>1.4550</td>
      <td>248.407</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11 Down</td>
      <td>1.5125</td>
      <td>239.704</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11 Down</td>
      <td>1.5375</td>
      <td>211.602</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11 Down</td>
      <td>1.5600</td>
      <td>173.626</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AB Root Beer</td>
      <td>3.7300</td>
      <td>384.838</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>AB Root Beer</td>
      <td>3.7700</td>
      <td>389.141</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AB Root Beer</td>
      <td>3.8125</td>
      <td>379.363</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AB Root Beer</td>
      <td>3.8425</td>
      <td>376.175</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Alpine Stream</td>
      <td>1.9975</td>
      <td>201.513</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Alpine Stream</td>
      <td>2.1375</td>
      <td>174.656</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Alpine Stream</td>
      <td>2.1500</td>
      <td>176.016</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Alpine Stream</td>
      <td>2.2275</td>
      <td>156.269</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Bright</td>
      <td>1.2725</td>
      <td>3790.724</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Bright</td>
      <td>1.2825</td>
      <td>2802.034</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Bright</td>
      <td>1.2900</td>
      <td>2656.430</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Crisp Clear</td>
      <td>1.3125</td>
      <td>184.459</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Crisp Clear</td>
      <td>1.3425</td>
      <td>173.047</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Crisp Clear</td>
      <td>1.4275</td>
      <td>138.466</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Crisp Clear</td>
      <td>1.4700</td>
      <td>140.921</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>DC Kola</td>
      <td>1.8900</td>
      <td>936.894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>DC Kola</td>
      <td>1.9150</td>
      <td>568.984</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>DC Kola</td>
      <td>1.9250</td>
      <td>512.329</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>DC Kola</td>
      <td>1.9325</td>
      <td>501.308</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Koala Kola</td>
      <td>2.4825</td>
      <td>1394.552</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Koala Kola</td>
      <td>2.5350</td>
      <td>1408.344</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Koala Kola</td>
      <td>2.5600</td>
      <td>1426.688</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Koala Kola</td>
      <td>2.5650</td>
      <td>1423.454</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Mr. Popper</td>
      <td>2.8475</td>
      <td>41.408</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Mr. Popper</td>
      <td>2.8925</td>
      <td>41.404</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Mr. Popper</td>
      <td>2.8950</td>
      <td>41.404</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Mr. Popper</td>
      <td>2.9000</td>
      <td>41.382</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Mr. Popper</td>
      <td>2.9850</td>
      <td>38.421</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Popsi Kola</td>
      <td>1.6725</td>
      <td>156.135</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Popsi Kola</td>
      <td>1.7125</td>
      <td>145.706</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Popsi Kola</td>
      <td>1.7275</td>
      <td>147.310</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Popsi Kola</td>
      <td>1.7500</td>
      <td>145.957</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from ortools.constraint_solver import pywrapcp as cp


solver = cp.Solver("nothing_fancy")
```


```python
x_array = []

"""x_array["Product", "Price"]"""

x_product = []
for i, product in enumerate(forecast_sales_r.drop_duplicates("Product")["Product"].values):
    t = []
    for j, price in enumerate(forecast_sales_r[forecast_sales_r["Product"]==product]["Product"].values):
        ## This in effect sets constraints on the domain
        t.append(solver.IntVar(0, 1, "x[%i,%i]" % (i, j)))
    x_array.extend(t)
    x_product.append(t)
```


```python
## More Specific Keys
x_array = []

"""x_array["Product", "Price"]"""

x_product = []
for i, product in enumerate(forecast_sales_r.drop_duplicates("Product")["Product"].values):
    t = []
    for j, price in enumerate(forecast_sales_r[forecast_sales_r["Product"]==product]["Cost Per Unit"].values):
        ## This in effect sets constraints on the domain
        t.append(solver.IntVar(0, 1, "x[%s,%s]" % (product, price)))
    x_array.extend(t) ## produces a flat array
    x_product.append(t)
```


```python
x_product
```




    [[x[11 Down,1.455](0 .. 1),
      x[11 Down,1.5125](0 .. 1),
      x[11 Down,1.5375](0 .. 1),
      x[11 Down,1.56](0 .. 1)],
     [x[AB Root Beer,3.73](0 .. 1),
      x[AB Root Beer,3.77](0 .. 1),
      x[AB Root Beer,3.8125](0 .. 1),
      x[AB Root Beer,3.8425](0 .. 1)],
     [x[Alpine Stream,1.9975](0 .. 1),
      x[Alpine Stream,2.1375](0 .. 1),
      x[Alpine Stream,2.15](0 .. 1),
      x[Alpine Stream,2.2275](0 .. 1)],
     [x[Bright,1.2725](0 .. 1), x[Bright,1.2825](0 .. 1), x[Bright,1.29](0 .. 1)],
     [x[Crisp Clear,1.3125](0 .. 1),
      x[Crisp Clear,1.3425](0 .. 1),
      x[Crisp Clear,1.4275](0 .. 1),
      x[Crisp Clear,1.47](0 .. 1)],
     [x[DC Kola,1.89](0 .. 1),
      x[DC Kola,1.915](0 .. 1),
      x[DC Kola,1.925](0 .. 1),
      x[DC Kola,1.9325](0 .. 1)],
     [x[Koala Kola,2.4825](0 .. 1),
      x[Koala Kola,2.535](0 .. 1),
      x[Koala Kola,2.56](0 .. 1),
      x[Koala Kola,2.565](0 .. 1)],
     [x[Mr. Popper,2.8475](0 .. 1),
      x[Mr. Popper,2.8925](0 .. 1),
      x[Mr. Popper,2.895](0 .. 1),
      x[Mr. Popper,2.9](0 .. 1),
      x[Mr. Popper,2.985](0 .. 1)],
     [x[Popsi Kola,1.6725](0 .. 1),
      x[Popsi Kola,1.7125](0 .. 1),
      x[Popsi Kola,1.7275](0 .. 1),
      x[Popsi Kola,1.75](0 .. 1)]]




```python
forecast_sales_r["Extra Sales Dollars"] = forecast_sales_r["Cost Per Unit"]*forecast_sales_r["extra_sales"]

forecast_sales_r["Investment Dollars"] = forecast_sales_r["marginal_investment"]*forecast_sales_r["Sales"]
```


```python
## I would change it to say largest sales dollars increase
## and then compare that against the investment 
## UNFORTUNATTELY - I Turned this into a knapsack problem 
```


```python
forecast_sales_r["Investment Dollars"] = forecast_sales_r["Investment Dollars"].abs()
```


```python
forecast_sales_r.head()
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
      <th>Product</th>
      <th>Cost Per Unit</th>
      <th>Sales</th>
      <th>marginal_investment</th>
      <th>extra_sales</th>
      <th>Extra Sales Dollars</th>
      <th>Investment Dollars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11 Down</td>
      <td>1.4550</td>
      <td>248.407</td>
      <td>-0.1050</td>
      <td>74.781</td>
      <td>108.806355</td>
      <td>26.082735</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11 Down</td>
      <td>1.5125</td>
      <td>239.704</td>
      <td>-0.0475</td>
      <td>66.078</td>
      <td>99.942975</td>
      <td>11.385940</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11 Down</td>
      <td>1.5375</td>
      <td>211.602</td>
      <td>-0.0225</td>
      <td>37.976</td>
      <td>58.388100</td>
      <td>4.761045</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11 Down</td>
      <td>1.5600</td>
      <td>173.626</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AB Root Beer</td>
      <td>3.7300</td>
      <td>384.838</td>
      <td>-0.1125</td>
      <td>8.663</td>
      <td>32.312990</td>
      <td>43.294275</td>
    </tr>
  </tbody>
</table>
</div>




```python
weights = []
weight_1 = list(forecast_sales_r["Investment Dollars"].values)

weights.append(weight_1)

capacities = [350]
values = list(forecast_sales_r["Extra Sales Dollars"].values)
```


```python
from ortools.algorithms import pywrapknapsack_solver

# Create the solver.
solver = pywrapknapsack_solver.KnapsackSolver(
  pywrapknapsack_solver.KnapsackSolver.
  KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
  'test')

solver.Init(values, weights, capacities)
computed_value = solver.Solve()

packed_items = [x for x in range(0, len(weights[0]))
                if solver.BestSolutionContains(x)]
packed_weights = [weights[0][i] for i in packed_items]

print("Packed items: ", packed_items)
print("Packed weights: ", packed_weights)
print("Total weight (same as total value): ", computed_value)
```

    Packed items:  [0, 1, 2, 5, 8, 9, 10, 12, 13, 15, 16, 19, 20, 21, 27, 28, 29, 30, 34]
    Packed weights:  [26.082735000000014, 11.385940000000028, 4.761044999999996, 28.212722499999938, 46.34799000000006, 15.71903999999999, 13.641240000000034, 66.33767000000027, 21.01525500000017, 29.052292500000004, 22.0634925, 39.817995000000145, 9.957220000000047, 3.842467500000032, 5.693599999999984, 3.829869999999989, 3.7263599999999912, 3.517469999999996, 3.314474999999995]
    Total weight (same as total value):  3220



```python
packed_items = [x for x in range(0, len(weights[0]))
                  if solver.BestSolutionContains(x)]

## This dataframe is the best selection, using the knapsack problem.
## Different costs differnt values 
## The knapsack problem does appear i a lot of places 
```


```python
forecast_sales_r[forecast_sales_r.index.isin(packed_items)]
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
      <th>Product</th>
      <th>Cost Per Unit</th>
      <th>Sales</th>
      <th>marginal_investment</th>
      <th>extra_sales</th>
      <th>Extra Sales Dollars</th>
      <th>Investment Dollars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11 Down</td>
      <td>1.4550</td>
      <td>248.407</td>
      <td>-0.1050</td>
      <td>74.781</td>
      <td>108.806355</td>
      <td>26.082735</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11 Down</td>
      <td>1.5125</td>
      <td>239.704</td>
      <td>-0.0475</td>
      <td>66.078</td>
      <td>99.942975</td>
      <td>11.385940</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11 Down</td>
      <td>1.5375</td>
      <td>211.602</td>
      <td>-0.0225</td>
      <td>37.976</td>
      <td>58.388100</td>
      <td>4.761045</td>
    </tr>
    <tr>
      <th>5</th>
      <td>AB Root Beer</td>
      <td>3.7700</td>
      <td>389.141</td>
      <td>-0.0725</td>
      <td>12.966</td>
      <td>48.881820</td>
      <td>28.212722</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Alpine Stream</td>
      <td>1.9975</td>
      <td>201.513</td>
      <td>-0.2300</td>
      <td>45.244</td>
      <td>90.374890</td>
      <td>46.347990</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Alpine Stream</td>
      <td>2.1375</td>
      <td>174.656</td>
      <td>-0.0900</td>
      <td>18.387</td>
      <td>39.302213</td>
      <td>15.719040</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Alpine Stream</td>
      <td>2.1500</td>
      <td>176.016</td>
      <td>-0.0775</td>
      <td>19.747</td>
      <td>42.456050</td>
      <td>13.641240</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Bright</td>
      <td>1.2725</td>
      <td>3790.724</td>
      <td>-0.0175</td>
      <td>1134.294</td>
      <td>1443.389115</td>
      <td>66.337670</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Bright</td>
      <td>1.2825</td>
      <td>2802.034</td>
      <td>-0.0075</td>
      <td>145.604</td>
      <td>186.737130</td>
      <td>21.015255</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Crisp Clear</td>
      <td>1.3125</td>
      <td>184.459</td>
      <td>-0.1575</td>
      <td>43.538</td>
      <td>57.143625</td>
      <td>29.052293</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Crisp Clear</td>
      <td>1.3425</td>
      <td>173.047</td>
      <td>-0.1275</td>
      <td>32.126</td>
      <td>43.129155</td>
      <td>22.063492</td>
    </tr>
    <tr>
      <th>19</th>
      <td>DC Kola</td>
      <td>1.8900</td>
      <td>936.894</td>
      <td>-0.0425</td>
      <td>435.586</td>
      <td>823.257540</td>
      <td>39.817995</td>
    </tr>
    <tr>
      <th>20</th>
      <td>DC Kola</td>
      <td>1.9150</td>
      <td>568.984</td>
      <td>-0.0175</td>
      <td>67.676</td>
      <td>129.599540</td>
      <td>9.957220</td>
    </tr>
    <tr>
      <th>21</th>
      <td>DC Kola</td>
      <td>1.9250</td>
      <td>512.329</td>
      <td>-0.0075</td>
      <td>11.021</td>
      <td>21.215425</td>
      <td>3.842468</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Mr. Popper</td>
      <td>2.8475</td>
      <td>41.408</td>
      <td>-0.1375</td>
      <td>2.987</td>
      <td>8.505482</td>
      <td>5.693600</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Mr. Popper</td>
      <td>2.8925</td>
      <td>41.404</td>
      <td>-0.0925</td>
      <td>2.983</td>
      <td>8.628327</td>
      <td>3.829870</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Mr. Popper</td>
      <td>2.8950</td>
      <td>41.404</td>
      <td>-0.0900</td>
      <td>2.983</td>
      <td>8.635785</td>
      <td>3.726360</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Mr. Popper</td>
      <td>2.9000</td>
      <td>41.382</td>
      <td>-0.0850</td>
      <td>2.961</td>
      <td>8.586900</td>
      <td>3.517470</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Popsi Kola</td>
      <td>1.7275</td>
      <td>147.310</td>
      <td>-0.0225</td>
      <td>1.353</td>
      <td>2.337308</td>
      <td>3.314475</td>
    </tr>
  </tbody>
</table>
</div>




```python

```




    [0, 1, 2, 5, 8, 9, 10, 12, 13, 15, 16, 19, 20, 21, 27, 28, 29, 30, 34]




```python
total_cost = solver.IntVar(0, 300, "total_cost")

solver.Add(total_cost <= 300)
```


```python
decision_builder = solver.Phase([x, y],
                                  solver.CHOOSE_FIRST_UNBOUND,
                                  solver.ASSIGN_MIN_VALUE)
```


```python
# Total cost
solver.Add(
  total_cost < solver.Sum(solver.ScalProd(x_array, forecast_sales_r["marginal_investment"].values)))
objective = solver.Minimize(total_cost, 1)
db = solver.Phase(x_array,
                solver.CHOOSE_FIRST_UNBOUND, # The VAR strategy
                solver.ASSIGN_MIN_VALUE)     # The Value strategy

# Create a solution collector.
collector = solver.LastSolutionCollector()
# Add decision variables
#collector.Add(x_array)

for i in range(num_workers):
    collector.Add(x_worker[i])
```


    ---------------------------------------------------------------------------

    NotImplementedError                       Traceback (most recent call last)

    <ipython-input-126-c28e31bfa220> in <module>()
          1 # Total cost
          2 solver.Add(
    ----> 3   total_cost < solver.Sum(solver.ScalProd(x_array, forecast_sales_r["marginal_investment"].values)))
          4 objective = solver.Minimize(total_cost, 1)
          5 db = solver.Phase(x_array,


    ~/.local/lib/python3.6/site-packages/ortools/constraint_solver/pywrapcp.py in ScalProd(self, *args)
        393 
        394     def ScalProd(self, *args) -> "operations_research::IntExpr *":
    --> 395         return _pywrapcp.Solver_ScalProd(self, *args)
        396 
        397     def MonotonicElement(self, values: 'operations_research::Solver::IndexEvaluator1', increasing: 'bool', index: 'IntVar') -> "operations_research::IntExpr *":


    NotImplementedError: Wrong number or type of arguments for overloaded function 'Solver_ScalProd'.
      Possible C/C++ prototypes are:
        operations_research::Solver::MakeScalProd(std::vector< operations_research::IntVar *,std::allocator< operations_research::IntVar * > > const &,std::vector< int64,std::allocator< int64 > > const &)
        operations_research::Solver::MakeScalProd(std::vector< operations_research::IntVar *,std::allocator< operations_research::IntVar * > > const &,std::vector< int,std::allocator< int > > const &)




```python
# Total cost
solver.Add(
  total_cost == solver.Sum( [solver.ScalProd(x_row, cost_row) for (x_row, cost_row) in zip(x_array, forecast_sales_r["marginal_investment"].values)]))
objective = solver.Minimize(total_cost, 1)
db = solver.Phase(x_array,
                solver.CHOOSE_FIRST_UNBOUND, # The VAR strategy
                solver.ASSIGN_MIN_VALUE)     # The Value strategy

# Create a solution collector.
collector = solver.LastSolutionCollector()
# Add decision variables
#collector.Add(x_array)

for i in range(num_workers):
    collector.Add(x_worker[i])
```


```python
## I only want to do two promotions for easy soda category
## for now I will just ignore this constraint, 
## one soda onl once - true
```


```python
max_prom
```




    {'Clear': 2, 'Dark': 2}




```python
for code, price, ranger in zip([ra[0] for ra in list(product_prices)],[ra[1] for ra in list(product_prices)],range(len([ra[1] for ra in list(product_prices)]))):
    print(code, price)
    select_price[code, ranger] = solver.IntVar(ub=price, lb=price, name='X')
```
