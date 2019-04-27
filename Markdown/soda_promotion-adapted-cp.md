
##   Remember to Restart Solver Often


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
forecast_sales = df_superbowl_original[["Product", "Cost Per Unit"]].copy()
forecast_sales["Sales"] = predicted

forecast_sales["Sales"] = (forecast_sales["Sales"]).astype(int)

forecast_sales["Cost Per Unit"] = (forecast_sales["Cost Per Unit"]  *1000).astype(int)

forecast_sales = forecast_sales.groupby(['Product','Cost Per Unit']).mean()
```


```python
forecast_sales.head()
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
      <th>Sales</th>
    </tr>
    <tr>
      <th>Product</th>
      <th>Cost Per Unit</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">11 Down</th>
      <th>1455</th>
      <td>243</td>
    </tr>
    <tr>
      <th>1512</th>
      <td>239</td>
    </tr>
    <tr>
      <th>1537</th>
      <td>215</td>
    </tr>
    <tr>
      <th>1560</th>
      <td>180</td>
    </tr>
    <tr>
      <th>AB Root Beer</th>
      <th>3730</th>
      <td>392</td>
    </tr>
  </tbody>
</table>
</div>




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
for j in range(num_tasks):
    ## This in effect sets constraints on the domain
    t.append(solver.IntVar(0, 1, "x[%i,%i]" % (i, j)))

```python
from ortools.constraint_solver import pywrapcp as cp

solver = cp.Solver("nothing_fancy")

```
from ortools.linear_solver import pywraplp  ## See this is Linear Programming

solver = pywraplp.Solver('SolveIntegerProblem',
                       pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)


### The thing is, once you use this the X_array looks wrong

```python
## More Specific Keys
forecast_sales_r =forecast_sales.reset_index()

x_array = []

"""x_array["Product", "Price"]"""
x_product = []
for i, product in enumerate(forecast_sales_r.drop_duplicates("Product")["Product"].values):
    t = []
    for j, price in enumerate(forecast_sales_r[forecast_sales_r["Product"]==product]["Cost Per Unit"].values):
        ## This in effect sets constraints on the domain
        t.append(solver.BoolVar("x[%s,%s]" % (product, price)))
    x_array.extend(t) ## produces a flat array
    x_product.append(t)
    
### Because it occurs in the same order you can also use x_product
```


```python
products = forecast_sales_r.drop_duplicates("Product")["Product"].values
```


```python
### This is the problems with strings, you can't call them like this.
### x_array["11 Down"]
```


```python
x_product[0]
```




    [x[11 Down,1455](0 .. 1),
     x[11 Down,1512](0 .. 1),
     x[11 Down,1537](0 .. 1),
     x[11 Down,1560](0 .. 1)]




```python
int(forecast_sales_r["Sales"].max()+1)
```




    3799




```python
forecast_sales_r["Revenue"] = forecast_sales_r["Sales"]*forecast_sales_r["Cost Per Unit"]

## x_product, x_array
investment = solver.IntVar(0, 350*(1000)*1, "investment")
sales = solver.IntVar(0, int(forecast_sales_r["Sales"].max()*len(forecast_sales_r)), 'sales')
revenue = solver.IntVar(0, int(forecast_sales_r["Revenue"].max()*len(forecast_sales_r)), 'revenue')

```


```python
# Total cost
solver.Add(sales == solver.Sum(x_array * forecast_sales_r.Sales))
```


```python
solver.Add(revenue == solver.Sum(forecast_sales.Sales[b,p] * p * 
                                 x_array[i] for i, (b,p) in enumerate(product_prices)))
```


```python
solver.Add(investment == 
                solver.Sum(max(0,forecast_sales.Sales[b,p] - 
                            forecast_sales.Sales[b,normal_price[b]]) *
                        normal_price[b] * x_array[i] 
                        for i, (b,p) in enumerate(product_prices)))
```


```python
## This one is important as it sets it in motion. 
obj = sales
#obj = revevue

### Eks nog steeds nie seker of jy die regte ding maximiseer nie
objective = solver.Maximize(obj, step=1)
```




    {'11 Down',
     'AB Root Beer',
     'Alpine Stream',
     'Bright',
     'Crisp Clear',
     'DC Kola',
     'Koala Kola',
     'Mr. Popper',
     'Popsi Kola'}




```python
db = solver.Phase(x_array,
                solver.CHOOSE_FIRST_UNBOUND,
                solver.ASSIGN_MIN_VALUE)

```


```python
collector = solver.LastSolutionCollector()
for i in range(len(x_array)):
    collector.Add(x_array[i])
collector.AddObjective(obj)

```


```python
collector.AddObjective(obj)
solver.Solve(db, [objective, collector])

```




    True




```python
x_array[1]
```




    x[11 Down,1512](0 .. 1)




```python
if collector.SolutionCount() > 0:
    best_solution = collector.SolutionCount() - 1
    print("Sales = ", collector.ObjectiveValue(best_solution))
    print()
    index = []
    for i in range(len(x_array)):
        if collector.Value(best_solution, x_array[i]) == 1:
            print('Item ', i)
            index.append(i)


# 12943
```

    Sales =  11982
    
    Item  3
    Item  7
    Item  11
    Item  12
    Item  13
    Item  14
    Item  18
    Item  22
    Item  26
    Item  30
    Item  31
    Item  35



```python
forecast_sales_r[forecast_sales_r.index.isin(index)]
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
      <th>Revenue</th>
      <th>Product_code</th>
      <th>Iteration_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>11 Down</td>
      <td>1560</td>
      <td>180</td>
      <td>280800</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AB Root Beer</td>
      <td>3842</td>
      <td>380</td>
      <td>1459960</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Alpine Stream</td>
      <td>2227</td>
      <td>149</td>
      <td>331823</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Bright</td>
      <td>1272</td>
      <td>3798</td>
      <td>4831056</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Bright</td>
      <td>1282</td>
      <td>2640</td>
      <td>3384480</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Bright</td>
      <td>1290</td>
      <td>2491</td>
      <td>3213390</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Crisp Clear</td>
      <td>1470</td>
      <td>211</td>
      <td>310170</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>22</th>
      <td>DC Kola</td>
      <td>1932</td>
      <td>412</td>
      <td>795984</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Koala Kola</td>
      <td>2565</td>
      <td>1508</td>
      <td>3868020</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Mr. Popper</td>
      <td>2900</td>
      <td>32</td>
      <td>92800</td>
      <td>7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Mr. Popper</td>
      <td>2985</td>
      <td>29</td>
      <td>86565</td>
      <td>7</td>
      <td>4</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Popsi Kola</td>
      <td>1750</td>
      <td>152</td>
      <td>266000</td>
      <td>8</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
forecast_sales_r[forecast_sales_r.index.isin(index)]["Sales"].sum()
```


```python
forecast_sales_r
```


```python
sol
```


```python
sales
```


```python
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
decision_builder = solver.Phase([x, y],
                                  solver.CHOOSE_FIRST_UNBOUND,
                                  solver.ASSIGN_MIN_VALUE)
```


```python
I tried my best to make sure that it only gets selected once,
well it did not work

because of some weird out of index issue, its shit

solver.Add((solver.Sum(x_array[i] for i, (b,p) in enumerate(product_prices) if (soda_family[b] == f) and (p != normal_price[b]) ) <= max_prom[f] for f in family))

soda

x_array[0]

x_product

unique_prod = len(forecast_sales_r.drop_duplicates("Product")["Product"].values)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()



forecast_sales_r["Product_code"] = le.fit_transform(forecast_sales_r["Product"])

forecast_sales_r["Iteration_code"] = forecast_sales_r.groupby('Product').cumcount()

forecast_sales_r.head()

for i in forecast_sales_r["Product_code"].unique():
    print(i)

## So at three works pretty well but need 4. 
## Dis moeilik om met die oneweridhedi the deal 
[solver.Add(solver.Sum(x_product[i][j] for i in forecast_sales_r["Product_code"].unique() ) <= 1) 
 for j in range(len(x_product[i]))]


## I have not ide whether this would work
## will have to work in  <= 1
dap = {}
for i, product in enumerate(forecast_sales_r.drop_duplicates("Product")["Product"].values):
    dap[product] = len(forecast_sales_r[forecast_sales_r["Product"]==product]["Cost Per Unit"].values)

## So at three works pretty well but need 4. 
## Dis moeilik om met die oneweridhedi the deal 
[solver.Add(solver.Sum(x_product[i][j] for i, product in enumerate(forecast_sales_r.drop_duplicates("Product")["Product"].values)) <= 1)
 for j in range(dap[product])]



## So at three works pretty well but need 4. 
## Dis moeilik om met die oneweridhedi the deal 
[solver.Add(solver.Sum(x_product[i][j] for i, product in enumerate(forecast_sales_r.drop_duplicates("Product")["Product"].values)) <= 1)
 for j in range(len(forecast_sales_r[forecast_sales_r["Product"]==product]["Cost Per Unit"]))]
```
